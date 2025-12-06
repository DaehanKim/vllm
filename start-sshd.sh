#!/usr/bin/env bash
set -euo pipefail

AUTH_FILE=/root/.ssh/authorized_keys
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Append provided public keys if present (RunPod uses SSH_PUBLIC_KEY for overrides).
for var in SSH_PUBLIC_KEY RUNPOD_PUBLIC_KEY PUBLIC_KEY; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        mkdir -p "$(dirname "$AUTH_FILE")"
        touch "$AUTH_FILE"
        if ! grep -qF "$val" "$AUTH_FILE"; then
            printf '%s\n' "$val" >> "$AUTH_FILE"
        fi
    fi
done

if [ -f "$AUTH_FILE" ]; then
    chmod 600 "$AUTH_FILE"
fi

# Harden to key-only auth for root.
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config || true
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config || true
sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config || true

if [ "$#" -gt 0 ]; then
    /usr/sbin/sshd
    exec "$@"
else
    exec /usr/sbin/sshd -D
fi
