import argparse
import csv
import json
import random
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ed_pl import TashkeelModel
from tashkeel_dataset import TashkeelDataset, BufferedBatchSampler, collate_packed
from tashkeel_tokenizer import TashkeelTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train packed CATT on one CUDA GPU.')
    parser.add_argument('--train-dir', default='dataset/train')
    parser.add_argument('--val-dir', default='dataset/val')
    parser.add_argument('--output-dir', default='/data/catt',
                        help='Exact directory for metrics, run configuration, and profiles.')
    parser.add_argument('--checkpoint-dir', default='/data/checkpoints',
                        help='Load the alphabetically last file and save checkpoints here.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--tokens-per-batch', type=int, default=16384,
                        help='Static token capacity; real token count must be below this limit.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-seq-len', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--log-every', type=int, default=100,
                        help='Flush buffered metrics every N records.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.tokens_per_batch < 2 or args.max_seq_len < 1:
        parser.error('Batch size and maximum sequence length must be positive; token capacity must exceed 1.')
    if args.log_every < 1:
        parser.error('--log-every must be positive.')
    return args


class GPUMetricSampler:
    def __init__(self):
        self.records = Queue()
        self.stop = Event()

    def __enter__(self):
        self.thread = Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.thread.join()

    def sample(self):
        while not self.stop.is_set():
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                result = subprocess.run(
                    ['nvidia-smi', '-i', '0',
                     '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm',
                     '--format=csv'],
                    capture_output=True, text=True, check=True, timeout=10,
                )
                rows = csv.reader(result.stdout.splitlines(), skipinitialspace=True)
                next(rows)  # CSV header includes units.
                for row in rows:
                    gpu_timestamp, utilization, memory, power, temperature, clock = row
                    values = [None if value.strip().startswith('[') else float(value.split()[0])
                              for value in (utilization, memory, power, temperature, clock)]
                    self.records.put(dict(
                        event='gpu_sample', timestamp=timestamp, gpu_timestamp=gpu_timestamp,
                        gpu_utilization_pct=values[0], gpu_memory_used_mib=values[1],
                        gpu_power_w=values[2], gpu_temperature_c=values[3], gpu_clock_sm_mhz=values[4],
                    ))
            except (OSError, subprocess.SubprocessError, ValueError, csv.Error, StopIteration) as exc:
                self.records.put(dict(event='gpu_sample_error', timestamp=timestamp, error=str(exc)))
            if self.stop.wait(10):
                break


class MetricLogger:
    def __init__(self, path, flush_every):
        self.path = path
        self.flush_every = flush_every
        self.records = []

    def __enter__(self):
        self.file = self.path.open('a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.flush()
        finally:
            self.file.close()

    def log(self, **metrics):
        self.records.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **{key: value.detach() if isinstance(value, torch.Tensor) else value
               for key, value in metrics.items()},
        })
        if len(self.records) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.records:
            return
        tensor_fields = [(record, key) for record in self.records
                         for key, value in record.items() if isinstance(value, torch.Tensor)]
        if tensor_fields:
            # Transfer buffered scalar metrics together without retaining autograd graphs.
            values = torch.stack([record[key] for record, key in tensor_fields]).cpu().tolist()
            for (record, key), value in zip(tensor_fields, values):
                record[key] = value
        self.file.writelines(json.dumps(record) + '\n' for record in self.records)
        self.file.flush()
        self.records.clear()


def to_cuda(batch):
    return {key: value.to('cuda', non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def forward(model, batch):
    return model(x=batch['src'], y=batch['trg'], cu_seqlens=batch['cu_seqlens'],
                 max_seqlen=batch['max_seqlen'], positions=batch['positions'])


@torch.no_grad()
def validate(model, loader, tokenizer):
    model.eval()
    loss_sum = torch.zeros((), device='cuda')
    token_count = distance = ref_length = 0
    for cpu_batch in tqdm(loader, desc='Validation'):
        batch = to_cuda(cpu_batch)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = forward(model=model, batch=batch)
            loss_sum += F.cross_entropy(logits, batch['labels'], reduction='sum')
        tokens = batch['tokens']
        token_count += tokens
        predictions = logits[:tokens].argmax(-1).cpu().split(cpu_batch['lengths'])
        sources = cpu_batch['src'][:tokens].split(cpu_batch['lengths'])
        targets = cpu_batch['labels'][:tokens].split(cpu_batch['lengths'])
        for src, pred, target in zip(sources, predictions, targets):
            hyp = tokenizer.decode(src.unsqueeze(0), pred.unsqueeze(0))[0]
            ref = tokenizer.decode(src.unsqueeze(0), target.unsqueeze(0))[0]
            der = tokenizer.compute_der(ref, hyp)
            distance += der['distance']
            ref_length += der['ref_length']
    return {'val_loss': loss_sum.item() / token_count,
            'val_der': distance / max(ref_length, 1),
            'val_der_distance': distance, 'val_der_ref_length': ref_length,
            'val_der_mode': 'teacher_forced'}

def estimate_tockens_budget(batch_size, lengths, percentile=99.9, trials=20_000, k=25):
    assert k < batch_size
    rng = np.random.default_rng(42)
    batches = rng.choice(lengths, size=(trials, batch_size))
    totals = batches.sum(axis=1)
    
    extra_batches = rng.choice(lengths, size=(trials, batch_size))
    removed = np.sort(batches, axis=1)[:, -k:].sum(axis=1)
    added = np.sort(extra_batches, axis=1)[:, :k].sum(axis=1)
    swapped_totals = totals - removed + added

    swapped_budget = int(np.percentile(swapped_totals, percentile, method="higher")) + 1
    return swapped_budget

def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = max(checkpoint_dir.iterdir())
    print(f'Loading checkpoint: {checkpoint_path}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    generator = torch.Generator().manual_seed(args.seed)

    run_dir = Path(args.output_dir)
    profile_dir = run_dir / 'profiles'
    profile_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'loaded_checkpoint': str(checkpoint_path),
              'device': torch.cuda.get_device_name(0),
              'precision': 'bf16-mixed', 'compile': {'backend': 'inductor', 'dynamic': False},
              'optimizer': {'name': 'AdamW', 'fused': True},
              'fallback_token_capacity': args.batch_size * args.max_seq_len + 1,
              'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
              'attention': 'torch.nn.attention.varlen.varlen_attn',
              'model': {'d_model': 512, 'n_layers': 3, 'n_heads': 16,
                        'drop_prob': 0.1, 'learnable_pos_emb': False}}
    (run_dir / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Run artifacts: {run_dir}')

    tokenizer = TashkeelTokenizer()
    train_dataset = TashkeelDataset(args.train_dir, tokenizer, args.max_seq_len, args.threshold)
    val_dataset = TashkeelDataset(args.val_dir, tokenizer, args.max_seq_len, args.threshold)
    if min(len(train_dataset), len(val_dataset)) == 0:
        raise ValueError('Training and validation datasets must each contain at least one sample.')
    tokens_per_batch = estimate_tockens_budget(batch_size=args.batch_size, lengths=train_dataset.lengths)
    print(f"{tokens_per_batch=}")

    collate_fn = partial(collate_packed, batch_size=args.batch_size, token_capacity=tokens_per_batch,
                         max_seq_len=args.max_seq_len,
                         src_pad_idx=tokenizer.letters_map['<PAD>'],
                         trg_pad_idx=tokenizer.tashkeel_map['<PAD>'])
    loader_options = dict(num_workers=args.num_workers,
                          collate_fn=collate_fn, pin_memory=True,
                          persistent_workers=args.num_workers > 0)
    train_sampler = BufferedBatchSampler(train_dataset.lengths, args.batch_size,
                                         tokens_per_batch, shuffle=True, generator=generator)
    val_sampler = BufferedBatchSampler(val_dataset.lengths, args.batch_size, tokens_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                             generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_options)

    model = TashkeelModel(tokenizer, max_seq_len=args.max_seq_len,
                         **config['model']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / 1.000001)
    start_epoch = global_step = 0
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'optimizer' in checkpoint:
        if checkpoint['config']['max_seq_len'] != args.max_seq_len:
            raise ValueError('--max-seq-len must match the resumed checkpoint.')
        model.load_state_dict(checkpoint['state_dict'])
        # Keep fused execution when resuming checkpoints from the previous optimizer.
        for group in checkpoint['optimizer']['param_groups']:
            group.update(fused=True, foreach=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch, global_step = checkpoint['epoch'], checkpoint['global_step']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        generator.set_state(checkpoint['loader_rng_state'])
    else:
        weights = checkpoint.get('state_dict', checkpoint)
        weights = {key.removeprefix('transformer.'): value for key, value in weights.items()}
        print(f'Pretrained weights: {model.transformer.load_state_dict(weights, strict=False)}')

    compiled_model = torch.compile(model, backend='inductor', dynamic=False)
    launch_steps = 0
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    def save_trace(profiler):
        # Do not label an incomplete capture as a two-step profile.
        if launch_steps >= 17:
            profiler.export_chrome_trace(str(profile_dir / 'train-steps-16-17.json'))

    with (
        torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=15, active=2, repeat=1),
            on_trace_ready=save_trace, record_shapes=True, profile_memory=True,
        ) as profiler,
        MetricLogger(path=run_dir / 'metrics.jsonl', flush_every=args.log_every) as logger,
        GPUMetricSampler() as gpu_sampler,
    ):
        for epoch in range(start_epoch + 1, args.epochs + 1):
            compiled_model.train()
            if 0 < launch_steps < 17:
                profiler.toggle_collection_dynamic(True, activities)
            train_loss_sum = torch.zeros((), device='cuda')
            train_tokens = 0
            for cpu_batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
                batch = to_cuda(cpu_batch)
                optimizer.zero_grad(set_to_none=True)
                lr = optimizer.param_groups[0]['lr']
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = forward(model=compiled_model, batch=batch)
                    loss = F.cross_entropy(logits, batch['labels'])
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1
                launch_steps += 1
                tokens = batch['tokens']
                train_loss_sum += loss.detach() * tokens
                train_tokens += tokens
                logger.log(event='train_step', epoch=epoch, step=global_step,
                           train_loss=loss, lr=lr, tokens=tokens,
                           token_capacity=batch['labels'].numel())
                while True:
                    try:
                        record = gpu_sampler.records.get_nowait()
                    except Empty:
                        break
                    logger.log(**record)
                if launch_steps <= 17:
                    profiler.step()

            logger.flush()
            # Validation/checkpoint work must not enter the two training-step trace,
            # including when a small dataset puts steps 16 and 17 in different epochs.
            if launch_steps < 17:
                profiler.toggle_collection_dynamic(False, activities)
            metrics = validate(compiled_model, val_loader, tokenizer)
            metrics['train_loss'] = train_loss_sum.item() / train_tokens
            logger.log(event='epoch_end', epoch=epoch, step=global_step, **metrics)
            logger.flush()
            checkpoint = {
                'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch,
                'global_step': global_step, 'config': config, 'metrics': metrics,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'python_rng_state': random.getstate(), 'loader_rng_state': generator.get_state(),
            }
            checkpoint_path = checkpoint_dir / f'train-epoch-{epoch:06d}.pt'
            temporary_path = checkpoint_path.with_suffix('.tmp')
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)
            for old_checkpoint in sorted(checkpoint_dir.glob('train-epoch-*.pt'))[:-20]:
                old_checkpoint.unlink()
            print(f'Saved {checkpoint_path}; val_loss={metrics["val_loss"]:.5f}, '
                  f'val_der={metrics["val_der"]:.5f}')
        if launch_steps < 17:
            logger.log(event='profile_incomplete', train_steps=launch_steps,
                       required_steps=17)


if __name__ == '__main__':
    main()
