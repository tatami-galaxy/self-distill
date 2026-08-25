# Syncing the checkpoint archive

The archive is stored in the public Hugging Face bucket:

- Web: <https://huggingface.co/buckets/Ujan/self-distill-checkpoints>
- URI: `hf://buckets/Ujan/self-distill-checkpoints`
- Source directory on the current server: `/mnt/data/ujan/self-distill`

## Install the Hugging Face CLI

Use a recent `huggingface_hub` release so that the `hf buckets` commands and
Xet transfer support are available:

```sh
python3 -m pip install --upgrade huggingface_hub
hf --version
```

Uploading requires a Hugging Face token with write access to the `Ujan`
namespace:

```sh
hf auth login
hf auth whoami
```

The bucket is public, so downloading does not require authentication.

## Upload or synchronize the local folder

Preview the changes without transferring anything:

```sh
hf buckets sync \
  /mnt/data/ujan/self-distill \
  hf://buckets/Ujan/self-distill-checkpoints \
  --dry-run
```

Upload new and changed files:

```sh
HF_XET_HIGH_PERFORMANCE=1 hf buckets sync \
  /mnt/data/ujan/self-distill \
  hf://buckets/Ujan/self-distill-checkpoints \
  --delete
```

The command compares the local and remote trees, skips unchanged files, and
can be resumed by running the same command again. Run it directly inside
`tmux` or `screen` for a long transfer so that the progress display remains
visible and the transfer survives a disconnected terminal.

Do not synchronize a checkpoint while it is being written. If training is
active during a sync, run the command again after training finishes.

By default, syncing only adds or updates remote files. Do not add `--delete`
unless remote files missing from the local folder should be permanently
deleted from the bucket.

To inspect the remote contents:

```sh
hf buckets list Ujan/self-distill-checkpoints --recursive --human-readable
```

## Download the bucket elsewhere

Install a recent `huggingface_hub` package on the destination machine, then
preview the download:

```sh
hf buckets sync \
  hf://buckets/Ujan/self-distill-checkpoints \
  /path/to/self-distill \
  --dry-run
```

Download the complete archive:

```sh
HF_XET_HIGH_PERFORMANCE=1 hf buckets sync \
  hf://buckets/Ujan/self-distill-checkpoints \
  /path/to/self-distill
```

Replace `/path/to/self-distill` with the desired destination. Ensure that the
destination filesystem has enough free space for the full archive. If the
download is interrupted, rerun the same command; files already synchronized
will be skipped.

After completion, run the download command once more with `--dry-run`. A fully
synchronized destination should report no downloads.
