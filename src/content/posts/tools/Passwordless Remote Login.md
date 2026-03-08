---
title: "Passwordless Remote Login"
published: 2026-01-30
description: "Passwordless Remote Login"
image: ""
tags: ["tools","Passwordless Remote Login"]
category: tools
draft: false
lang: ""
---

# **I. SSH Passwordless Login — Key-Based Authentication**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> The goal is to log in to a remote server (e.g., <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">spiedie.binghamton.edu</code>) from Mac/Windows/Linux <strong>without entering a password</strong>, and to make VS Code Remote-SSH connections more stable. This is achieved by replacing password-based authentication with <strong>public/private key authentication (公钥/私钥认证)</strong>.
</div>

---

## 1. Core Concept: How SSH Passwordless Login Works

SSH "passwordless login" does not skip identity verification — it replaces password-based authentication with key-based authentication.

### 1) Two Key Files

- <span style="color:#E8600A;font-weight:700">Private Key (私钥)</span>: stored on your local machine
  Example: `~/.ssh/id_ed25519`
  <span style="color:#C0392B;font-weight:600">Must never be leaked or shared</span>

- <span style="color:#E8600A;font-weight:700">Public Key (公钥)</span>: can be sent to the server
  Example: `~/.ssh/id_ed25519.pub`
  Safe to share openly / copy to servers

### 2) How Does the Server Remember You?

The server stores your public key in:

```bash
~/.ssh/authorized_keys
```

Once added, the server will allow anyone who holds the corresponding private key to log in.

### 3) What Happens During Authentication (Simplified Flow)

1. You initiate a connection: `ssh user@host`
2. The server looks up your public key in `authorized_keys`
3. The server sends a random **challenge**
4. Your machine signs the challenge using your **private key** (the private key never leaves your machine)
5. The server verifies the signature using your **public key**
6. Verification passes → Login succeeds (no password prompt)

---

## 2. Step-by-Step Setup

The following examples use:

- Username: `xli49`
- Host: `spiedie.binghamton.edu`

### 1) Step 1: Generate an SSH Key on Your Local Machine

Check whether a key already exists:

```bash
ls ~/.ssh/id_ed25519 ~/.ssh/id_rsa 2>/dev/null
```

If not, generate one (ed25519 recommended):

```bash
ssh-keygen -t ed25519 -C "xli49@spiedie.binghamton.edu"
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> For a completely password-free experience, press Enter to leave the passphrase empty. For better security, set a passphrase and use it with Keychain / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ssh-agent</code>.</div>

### 2) Step 2: Copy the Public Key to the Server

✅ Recommended (when `ssh-copy-id` is available):

```bash
ssh-copy-id xli49@spiedie.binghamton.edu
```

Enter your password once — that's the last time.

#### Manual Method (if `ssh-copy-id` is not available)

```bash
cat ~/.ssh/id_ed25519.pub | ssh xli49@spiedie.binghamton.edu \
"mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
```

### 3) Step 3: Test the Passwordless Login

```bash
ssh xli49@spiedie.binghamton.edu
```

✅ If no password prompt appears → setup successful.

---

## 3. Configuring `~/.ssh/config` (Recommended)

Edit the config file:

```bash
nano ~/.ssh/config
```

Recommended configuration (using the full hostname as the `Host`):

```sshconfig
Host spiedie.binghamton.edu
    HostName spiedie.binghamton.edu
    User xli49
    ServerAliveInterval 300
    ServerAliveCountMax 120
```

After saving, connect with just:

```bash
ssh spiedie.binghamton.edu
```

---

## 4. When Is `IdentityFile` Needed?

### 1) Usually Not Required

SSH automatically tries the following keys in order:

- `~/.ssh/id_ed25519`
- `~/.ssh/id_rsa`
- Any keys already loaded into `ssh-agent`

If your key is in one of these default locations, you typically do not need to specify `IdentityFile`.

### 2) When You Must Specify `IdentityFile`

- <span style="color:#2980B9">You have multiple keys</span> and SSH might pick the wrong one
- <span style="color:#2980B9">Your key is not in a default path</span>
- <span style="color:#2980B9">The server requires a specific key</span>

Example:

```sshconfig
Host spiedie.binghamton.edu
    HostName spiedie.binghamton.edu
    User xli49
    IdentityFile ~/.ssh/id_ed25519
```

---

## 5. Preventing VS Code Remote-SSH Disconnections (KeepAlive)

Common causes of disconnection:

- The school network or firewall clears long-idle connections
- The remote session is considered idle when there is no output for an extended period

Solution: enable SSH heartbeat packets (KeepAlive):

```sshconfig
ServerAliveInterval 30
ServerAliveCountMax 120
```

What this means:

- Send a heartbeat packet every <span style="color:#E8600A;font-weight:700">30 seconds</span>
- Allow up to <span style="color:#E8600A;font-weight:700">120 consecutive non-responses</span> (~1 hour) before disconnecting

---

## 6. Troubleshooting

### 1) Check Which Key SSH Is Using

```bash
ssh -v xli49@spiedie.binghamton.edu
```

Look for lines like:

- `Offering public key: ...`
- `Authentication succeeded`

### 2) Check the Default Shell on the Server

Run remotely to rule out misconfiguration:

```bash
echo $SHELL
getent passwd xli49 | cut -d: -f7
which zsh
```

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Generate a key with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ssh-keygen -t ed25519</code>, copy it to the server once with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ssh-copy-id</code>, add <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ServerAliveInterval 30</code> to <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/.ssh/config</code>, and you'll never type a password or suffer a dropped VS Code connection again.</div>