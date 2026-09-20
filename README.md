# Inkling deployment SSD: field update, 2026-09-19

Temporary download branch (not part of the source tree; safe to delete once used).
The contents are built from `feat/inkling-multistream` (PR #159), `deploy/inkling-fleet/`.

On a Linux machine with the SSD plugged in:

    curl -fLO https://raw.githubusercontent.com/labscommunity/cascadia/inkling-ssd-update-20260919/inkling-ssd-update.tar.gz
    curl -fLO https://raw.githubusercontent.com/labscommunity/cascadia/inkling-ssd-update-20260919/SHA256SUMS
    sha256sum -c SHA256SUMS
    tar xzf inkling-ssd-update.tar.gz && cd inkling-ssd-update
    sudo ./apply-update.sh /media/$USER/<ssd>      # the folder that holds inkling-deploy/ and inkling/

`UPDATE.md` inside the archive says what it changes and why.
