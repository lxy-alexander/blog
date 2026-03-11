---
title: "RLHF"
published: 2026-03-10
description: "RLHF"
image: ""
tags: ["vllm","RLHF"]
category: vllm
draft: false
lang: ""
---

# **I. RLHF 完整学习手册**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<span style="color:#E8600A;font-weight:700">RLHF (Reinforcement Learning from Human Feedback)</span> is a technique that aligns language models with human preferences through a three-stage pipeline: <span style="color:#2980B9">Supervised Fine-Tuning (SFT)</span>, <span style="color:#2980B9">Reward Modeling (RM)</span>, and <span style="color:#2980B9">Reinforcement Learning (RL)</span>. 
</div>

## 1. Stage 1: Supervised Fine-Tuning (SFT)

### 1) What is SFT?

SFT is the process of taking a base pre-trained language model and fine-tuning it on high-quality human demonstrations of desired behavior.

### 2) Mathematical Formulation

$$\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \log P_{\theta}(y_t | x, y_{<t})$$

### 3) Symbol Breakdown

| Symbol              | Meaning                    | 中文解释              | Example Value  |
| ------------------- | -------------------------- | --------------------- | -------------- |
| $\mathcal{L}_{SFT}$ | SFT loss value             | SFT损失值             | 2.34           |
| $T$                 | Length of response         | 回答长度              | 10 tokens      |
| $P_{\theta}(y_t)$   | Probability of token $y_t$ | 模型预测第t个词的概率 | 0.85           |
| $x$                 | Input prompt               | 输入提示              | "What is 2+2?" |
| $y_{<t}$            | Previous tokens            | 之前生成的所有词      | "2+2="         |
| $\theta$            | Model parameters           | 模型参数              | -              |

### 4) Numerical Example

```python
# Example: SFT for a single response
prompt = "What is 2+2?"
correct_response = "4"

# Step-by-step calculation
print("SFT Loss Calculation Example")
print("="*50)

# Token probabilities at each position (simplified)
probabilities = {
    '4': 0.85,     # Model thinks 85% chance of "4"
    'five': 0.10,  # 10% chance of "five"
    '3': 0.03,     # 3% chance of "3"
    '2': 0.02      # 2% chance of "2"
}

correct_prob = probabilities['4']
loss = -np.log(correct_prob)

print(f"Correct token probability: {correct_prob:.2f}")
print(f"Loss = -log({correct_prob:.2f}) = {loss:.4f}")
print(f"If model was very confident (0.99): loss = {-np.log(0.99):.4f}")
print(f"If model was uncertain (0.40): loss = {-np.log(0.40):.4f}")

# Output:
# Correct token probability: 0.85
# Loss = -log(0.85) = 0.1625
# If model was very confident (0.99): loss = 0.0101
# If model was uncertain (0.40): loss = 0.9163
```

### 5) Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SFTPhase:
    """
    Stage 1: Supervised Fine-Tuning
    """
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        
    def compute_loss(self, prompt_ids, response_ids):
        """
        Compute SFT loss for one example
        
        Args:
            prompt_ids: tokenized prompt [prompt_len]
            response_ids: tokenized response [response_len]
        """
        # Concatenate prompt and response
        input_ids = torch.cat([prompt_ids, response_ids])
        
        # Forward pass
        logits = self.model(input_ids)  # [seq_len, vocab_size]
        
        # We only care about predicting response tokens
        # Shift so that logits[t] predicts response_ids[t]
        response_logits = logits[prompt_ids.shape[0]-1:-1]  # [response_len, vocab_size]
        
        # Compute cross-entropy loss
        # loss = -∑ log P(y_t | x, y_<t)
        loss = self.criterion(
            response_logits.view(-1, response_logits.size(-1)),
            response_ids.view(-1)
        )
        
        # Manual calculation for understanding
        with torch.no_grad():
            probs = F.softmax(response_logits, dim=-1)
            correct_probs = probs.gather(1, response_ids.unsqueeze(1)).squeeze()
            manual_loss = -torch.log(correct_probs).mean()
            
            print(f"\nSFT Step Details:")
            for i, (prob, token) in enumerate(zip(correct_probs, response_ids)):
                print(f"  Token {i+1}: P={prob:.3f}, loss contribution={-torch.log(prob):.3f}")
            print(f"  Average loss = {manual_loss:.4f}")
        
        return loss
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">SFT Summary: </span>SFT teaches the model to imitate good examples. Loss = negative log probability of correct tokens. Lower loss = better imitation.</div>

## 2. Stage 2: Reward Modeling (RM)

### 1) What is Reward Modeling?

Reward modeling trains a separate model to predict human preferences. Given two responses to the same prompt, humans say which one is better. The reward model learns to assign higher scores to preferred responses.

### 2) Mathematical Formulation

The core of reward modeling is the **Bradley-Terry model**:

$$P(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}$$

The loss function is the negative log-likelihood:

$$\mathcal{L}_{RM} = -\log \left( \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} \right)$$

### 3) Symbol Breakdown

| Symbol                  | Meaning                                                     | 中文解释       | Example Value           |
| ----------------------- | ----------------------------------------------------------- | -------------- | ----------------------- |
| $P(y_w \succ y_l \| x)$ | Probability human prefers $y_w$ over $y_l$ given prompt $x$ | 人类偏好概率   | 0.85                    |
| $y_w$                   | Preferred response (winner)                                 | 被偏好的回答   | "The answer is 4"       |
| $y_l$                   | Less preferred response (loser)                             | 不被偏好的回答 | "I think it's 5"        |
| $x$                     | Input prompt                                                | 输入提示       | "What is 2+2?"          |
| $r(x, y)$               | Reward model score for (prompt, response)                   | 奖励分数       | 2.5                     |
| $\exp()$                | Exponential function                                        | 指数函数       | $e^{2.5} \approx 12.18$ |

### 4) Numerical Example

```python
def reward_modeling_example():
    """
    Numerical example of Bradley-Terry loss calculation
    """
    print("Reward Modeling Example")
    print("="*50)
    
    # Example 1: Good reward model (correctly ranks responses)
    print("\nCase 1: Good reward model")
    r_w = 2.5  # Score for preferred response
    r_l = 1.0  # Score for less preferred
    
    exp_w = np.exp(r_w)
    exp_l = np.exp(r_l)
    prob_w = exp_w / (exp_w + exp_l)
    loss = -np.log(prob_w)
    
    print(f"  Preferred score: {r_w}")
    print(f"  Less preferred score: {r_l}")
    print(f"  exp(w) = e^{r_w:.1f} = {exp_w:.2f}")
    print(f"  exp(l) = e^{r_l:.1f} = {exp_l:.2f}")
    print(f"  P(preferred) = {exp_w:.2f} / ({exp_w:.2f} + {exp_l:.2f}) = {prob_w:.4f}")
    print(f"  Loss = -log({prob_w:.4f}) = {loss:.4f}")
    
    # Example 2: Bad reward model (wrongly ranks)
    print("\nCase 2: Bad reward model (scores reversed)")
    r_w = 1.0  # Preferred gets lower score! (wrong)
    r_l = 2.5  # Less preferred gets higher score
    
    exp_w = np.exp(r_w)
    exp_l = np.exp(r_l)
    prob_w = exp_w / (exp_w + exp_l)
    loss = -np.log(prob_w)
    
    print(f"  Preferred score: {r_w} (should be higher!)")
    print(f"  Less preferred score: {r_l}")
    print(f"  P(preferred) = {exp_w:.2f} / ({exp_w:.2f} + {exp_l:.2f}) = {prob_w:.4f}")
    print(f"  Loss = -log({prob_w:.4f}) = {loss:.4f}")
    print(f"  Loss is much larger → model learns to fix this")
    
    # Example 3: Marginal difference
    print("\nCase 3: Small preference (scores close)")
    r_w = 2.1
    r_l = 2.0
    
    exp_w = np.exp(r_w)
    exp_l = np.exp(r_l)
    prob_w = exp_w / (exp_w + exp_l)
    loss = -np.log(prob_w)
    
    print(f"  Preferred score: {r_w}")
    print(f"  Less preferred score: {r_l}")
    print(f"  Difference: {r_w - r_l:.1f}")
    print(f"  P(preferred) = {prob_w:.4f}")
    print(f"  Loss = {loss:.4f} (smaller, because difference is subtle)")
    
# Run example
reward_modeling_example()
```

### 5) Simplified Form

The Bradley-Terry loss can be simplified to:

$$\mathcal{L}_{RM} = -\log(\sigma(r(x, y_w) - r(x, y_l)))$$

Where $\sigma$ is the sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$

### 6) Code Implementation

```python
class RewardModelPhase:
    """
    Stage 2: Reward Modeling with Bradley-Terry loss
    """
    
    def __init__(self, reward_model):
        self.reward_model = reward_model
        
    def bradley_terry_loss(self, reward_w, reward_l):
        """
        Compute Bradley-Terry loss
        
        Formula: L = -log( sigmoid(reward_w - reward_l) )
        
        Args:
            reward_w: scores for preferred responses [batch_size]
            reward_l: scores for less preferred responses [batch_size]
        """
        # Difference in scores
        diff = reward_w - reward_l
        
        # Sigmoid = 1 / (1 + exp(-diff))
        prob_w_better = torch.sigmoid(diff)
        
        # Negative log likelihood
        loss = -torch.log(prob_w_better + 1e-8)
        
        return loss.mean()
    
    def step_by_step_calculation(self, reward_w, reward_l):
        """
        Show step-by-step calculation for understanding
        """
        print("\nBradley-Terry Loss Step-by-Step")
        print("-"*40)
        
        # Step 1: Raw scores
        print(f"Step 1: Raw scores")
        print(f"  r_w = {reward_w:.2f} (preferred)")
        print(f"  r_l = {reward_l:.2f} (less preferred)")
        
        # Step 2: Difference
        diff = reward_w - reward_l
        print(f"\nStep 2: Difference")
        print(f"  diff = r_w - r_l = {reward_w:.2f} - {reward_l:.2f} = {diff:.2f}")
        
        # Step 3: Exponential
        exp_neg_diff = np.exp(-diff)
        print(f"\nStep 3: Sigmoid calculation")
        print(f"  σ(diff) = 1 / (1 + e^(-diff))")
        print(f"  e^(-diff) = e^(-{diff:.2f}) = {exp_neg_diff:.4f}")
        
        # Step 4: Probability
        prob = 1 / (1 + exp_neg_diff)
        print(f"  P(preferred) = 1 / (1 + {exp_neg_diff:.4f}) = {prob:.4f}")
        
        # Step 5: Loss
        loss = -np.log(prob)
        print(f"\nStep 5: Loss")
        print(f"  L = -log(P) = -log({prob:.4f}) = {loss:.4f}")
        
        return loss
    
    def train_step(self, batch):
        """
        One training step with multiple comparisons
        """
        prompts, responses_w, responses_l = batch
        
        # Get reward scores
        reward_w = self.reward_model(prompts, responses_w)
        reward_l = self.reward_model(prompts, responses_l)
        
        # Compute loss
        loss = self.bradley_terry_loss(reward_w, reward_l)
        
        # Metrics for monitoring
        with torch.no_grad():
            accuracy = (reward_w > reward_l).float().mean()
            margin = (reward_w - reward_l).mean()
            
        return loss, {
            'accuracy': accuracy.item(),
            'margin': margin.item(),
            'reward_w_mean': reward_w.mean().item(),
            'reward_l_mean': reward_l.mean().item()
        }


# Numerical verification
rm_phase = RewardModelPhase(None)

# Test cases
test_cases = [
    (2.5, 1.0, "Strong preference"),
    (2.1, 2.0, "Weak preference"), 
    (1.0, 2.5, "Wrong order (high loss)"),
]

for r_w, r_l, desc in test_cases:
    print(f"\n{desc}")
    rm_phase.step_by_step_calculation(r_w, r_l)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">RM Summary: </span>Reward model learns to assign higher scores to preferred responses. Loss = -log σ(r_w - r_l). When r_w > r_l, loss is small; when r_w < r_l, loss is large.</div>

## 3. Stage 3: Reinforcement Learning (PPO)

### 1) What is PPO in RLHF?

PPO (Proximal Policy Optimization) fine-tunes the language model to maximize reward from the reward model, while staying close to the original SFT model to prevent reward hacking.

### 2) The Complete PPO Objective

$$\mathcal{L}^{PPO} = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) - \beta D_{KL}(\pi_{\theta} || \pi_{ref})]$$

### 3) Symbol Breakdown

| Symbol         | Meaning                                                      | 中文解释       | Example Value |
| -------------- | ------------------------------------------------------------ | -------------- | ------------- |
| $r_t(\theta)$  | Probability ratio $\frac{\pi_{\theta}(y_t\|x)}{\pi_{old}(y_t\|x)}$ | 新旧策略概率比 | 1.2           |
| $A_t$          | Advantage at step t                                          | 优势函数       | 1.5           |
| $\epsilon$     | Clip range (usually 0.2)                                     | 裁剪范围       | 0.2           |
| $\beta$        | KL penalty coefficient                                       | KL惩罚系数     | 0.01          |
| $D_{KL}$       | KL divergence                                                | KL散度         | 0.05          |
| $\pi_{\theta}$ | Current policy                                               | 当前策略       | -             |
| $\pi_{ref}$    | Reference policy (SFT model)                                 | 参考策略       | -             |

### 4) Understanding Each Component

#### A. Probability Ratio $r_t(\theta)$

$$r_t(\theta) = \frac{\pi_{\theta}(y_t|x)}{\pi_{old}(y_t|x)}$$

```python
def probability_ratio_example():
    """
    Numerical examples of probability ratio
    """
    print("Probability Ratio Examples")
    print("="*50)
    
    # Case 1: Same probability
    p_old = 0.5
    p_new = 0.5
    ratio = p_new / p_old
    print(f"\nCase 1: No change")
    print(f"  π_old = {p_old}, π_new = {p_new}")
    print(f"  r = {p_new} / {p_old} = {ratio}")
    
    # Case 2: Increased probability (good if action was good)
    p_old = 0.3
    p_new = 0.6
    ratio = p_new / p_old
    print(f"\nCase 2: Increased probability (2x)")
    print(f"  π_old = {p_old}, π_new = {p_new}")
    print(f"  r = {p_new} / {p_old} = {ratio}")
    
    # Case 3: Decreased probability (good if action was bad)
    p_old = 0.8
    p_new = 0.4
    ratio = p_new / p_old
    print(f"\nCase 3: Decreased probability (0.5x)")
    print(f"  π_old = {p_old}, π_new = {p_new}")
    print(f"  r = {p_new} / {p_old} = {ratio}")
    
    # In log space (more common in code)
    log_p_old = np.log(p_old)
    log_p_new = np.log(p_new)
    log_ratio = log_p_new - log_p_old
    print(f"\nIn log space:")
    print(f"  log π_old = {log_p_old:.3f}")
    print(f"  log π_new = {log_p_new:.3f}")
    print(f"  log r = {log_ratio:.3f}, r = {np.exp(log_ratio):.3f}")
```

#### B. Advantage $A_t$

Advantage measures how much better an action is compared to average:

$$A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

```python
def advantage_example():
    """
    Numerical examples of advantage calculation
    """
    print("\nAdvantage Examples")
    print("="*50)
    
    # Simplified: In RLHF, often just use reward - baseline
    print("\nSimplified Advantage:")
    
    # Case 1: Good action (above average)
    reward = 2.0
    baseline = 1.0
    advantage = reward - baseline
    print(f"\nCase 1: Good action")
    print(f"  Reward: {reward}")
    print(f"  Baseline (average): {baseline}")
    print(f"  Advantage = {reward} - {baseline} = {advantage} (positive)")
    
    # Case 2: Bad action (below average)
    reward = 0.5
    baseline = 1.0
    advantage = reward - baseline
    print(f"\nCase 2: Bad action")
    print(f"  Reward: {reward}")
    print(f"  Baseline: {baseline}")
    print(f"  Advantage = {reward} - {baseline} = {advantage} (negative)")
    
    # Case 3: Average action
    reward = 1.0
    baseline = 1.0
    advantage = reward - baseline
    print(f"\nCase 3: Average action")
    print(f"  Advantage = {advantage} (no change)")
```

#### C. The Clip Function

$$\text{clip}(r, 1-\epsilon, 1+\epsilon)$$

```python
def clip_example(epsilon=0.2):
    """
    Demonstrate the clipping function
    """
    print(f"\nClipping Function (ε={epsilon})")
    print("="*50)
    
    test_ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    
    print(f"\n{'Ratio':<8} {'Clipped':<10} {'Within range?':<12}")
    print("-"*40)
    
    for r in test_ratios:
        clipped = np.clip(r, 1-epsilon, 1+epsilon)
        within = (1-epsilon <= r <= 1+epsilon)
        print(f"{r:<8.2f} {clipped:<10.2f} {str(within):<12}")
    
    print(f"\nExplanation:")
    print(f"  Ratios within [{1-epsilon:.1f}, {1+epsilon:.1f}] are kept as is")
    print(f"  Ratios outside are clipped to the boundary")
    print(f"  This prevents too large updates")
```

### 5) Complete PPO Numerical Example

```python
def complete_ppo_example():
    """
    Complete walkthrough of PPO loss calculation with numbers
    """
    print("\n" + "="*60)
    print("COMPLETE PPO LOSS CALCULATION EXAMPLE")
    print("="*60)
    
    # Given values
    epsilon = 0.2
    advantage = 1.5  # Positive advantage (good action)
    
    print(f"\nInitial values:")
    print(f"  ε = {epsilon}")
    print(f"  A = {advantage} (positive = good action)")
    
    # Test different probability ratios
    ratios = [0.6, 0.8, 1.0, 1.2, 1.4]
    
    print(f"\n{'Ratio':<8} {'Unclipped':<12} {'Clipped':<10} {'Loss1':<10} {'Loss2':<10} {'Final Loss':<10}")
    print("-"*70)
    
    for r in ratios:
        # Unclipped objective: r * A
        unclipped = r * advantage
        
        # Clipped objective: clip(r) * A
        clipped_r = np.clip(r, 1-epsilon, 1+epsilon)
        clipped = clipped_r * advantage
        
        # Loss = -min(unclipped, clipped) (negative because we minimize)
        # In code: loss = -min(r*A, clip(r)*A)
        final_objective = min(unclipped, clipped)
        loss = -final_objective
        
        print(f"{r:<8.2f} {unclipped:<12.2f} {clipped:<10.2f} "
              f"{-unclipped:<10.2f} {-clipped:<10.2f} {loss:<10.2f}")
    
    print(f"\nExplanation for r=1.4 (too high):")
    print(f"  Unclipped = 1.4 * 1.5 = 2.1")
    print(f"  Clipped = min(1.4, 1.2) * 1.5 = 1.8")
    print(f"  Final = min(2.1, 1.8) = 1.8 (clipped version)")
    print(f"  Loss = -1.8 (we use the clipped version to be safe)")
```

### 6) KL Penalty

$$D_{KL}(P || Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}$$

```python
def kl_penalty_example():
    """
    Numerical example of KL divergence
    """
    print("\nKL Divergence Example")
    print("="*50)
    
    # Vocabulary of 3 words
    vocab = ['apple', 'banana', 'cherry']
    
    # Reference model distribution (SFT model)
    ref_probs = np.array([0.5, 0.3, 0.2])
    
    # Case 1: Policy identical to reference
    policy1 = np.array([0.5, 0.3, 0.2])
    
    # Case 2: Policy slightly different
    policy2 = np.array([0.6, 0.25, 0.15])
    
    # Case 3: Policy very different
    policy3 = np.array([0.9, 0.05, 0.05])
    
    print(f"\n{'Word':<10} {'Ref P':<8} {'Policy1':<8} {'Policy2':<8} {'Policy3':<8}")
    print("-"*45)
    
    for i, word in enumerate(vocab):
        print(f"{word:<10} {ref_probs[i]:<8.2f} {policy1[i]:<8.2f} "
              f"{policy2[i]:<8.2f} {policy3[i]:<8.2f}")
    
    def compute_kl(p, q):
        return np.sum(p * np.log(p / q))
    
    kl1 = compute_kl(policy1, ref_probs)
    kl2 = compute_kl(policy2, ref_probs)
    kl3 = compute_kl(policy3, ref_probs)
    
    print(f"\nKL Divergence Results:")
    print(f"  KL(P1 || Ref) = {kl1:.4f} (identical → 0)")
    print(f"  KL(P2 || Ref) = {kl2:.4f}")
    print(f"  KL(P3 || Ref) = {kl3:.4f} (very different → larger)")
    
    # With KL penalty
    beta = 0.01
    reward = 2.0
    
    print(f"\nWith KL Penalty (β={beta}):")
    print(f"  Raw reward: {reward}")
    print(f"  Penalized reward (P2): {reward - beta * kl2:.4f}")
    print(f"  Penalized reward (P3): {reward - beta * kl3:.4f}")
    print(f"  P3's reward is reduced more because it drifted further")
```

### 7) Full PPO Implementation

```python
class PPOPolicy:
    """
    Complete PPO implementation with all components
    """
    
    def __init__(self, policy_model, ref_model, reward_model, epsilon=0.2, beta=0.01):
        self.policy = policy_model  # π_θ (trainable)
        self.ref_model = ref_model  # π_ref (frozen)
        self.reward_model = reward_model  # r(x,y) (frozen)
        self.epsilon = epsilon
        self.beta = beta
        
    def compute_kl(self, logits_current, logits_ref):
        """
        Compute KL divergence between current and reference policy
        """
        probs_current = F.softmax(logits_current, dim=-1)
        probs_ref = F.softmax(logits_ref, dim=-1)
        
        log_ratio = torch.log(probs_current + 1e-10) - torch.log(probs_ref + 1e-10)
        kl = (probs_current * log_ratio).sum(dim=-1)
        
        return kl.mean()
    
    def ppo_loss(self, prompts, responses, old_log_probs, advantages):
        """
        Compute full PPO loss with clipping and KL penalty
        """
        # Get current policy log probs
        log_probs_current = self.policy.get_log_probs(prompts, responses)
        
        # Probability ratio r_t(θ) = exp(log_probs_current - old_log_probs)
        log_ratio = log_probs_current - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # Unclipped objective
        pg_loss1 = -ratio * advantages
        
        # Clipped objective
        pg_loss2 = -torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        
        # PPO clipping loss
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # KL penalty
        with torch.no_grad():
            logits_current = self.policy(prompts)
            logits_ref = self.ref_model(prompts)
            kl = self.compute_kl(logits_current, logits_ref)
        
        # Total loss
        total_loss = pg_loss + self.beta * kl
        
        return total_loss, {
            'pg_loss': pg_loss.item(),
            'kl': kl.item(),
            'mean_ratio': ratio.mean().item(),
            'clip_frac': (torch.abs(ratio - 1) > self.epsilon).float().mean().item()
        }
    
    def numerical_example_step(self):
        """
        Step-by-step numerical example of PPO update
        """
        print("\n" + "="*60)
        print("PPO LOSS NUMERICAL EXAMPLE")
        print("="*60)
        
        # Simulated values for one token
        old_prob = 0.3      # π_old
        new_prob = 0.6      # π_θ
        advantage = 1.5     # A (positive)
        epsilon = 0.2
        beta = 0.01
        kl = 0.15           # Example KL
        
        print(f"\nInput values:")
        print(f"  π_old = {old_prob:.2f}")
        print(f"  π_θ = {new_prob:.2f}")
        print(f"  A = {advantage:.2f}")
        print(f"  ε = {epsilon}")
        print(f"  β = {beta}")
        print(f"  KL = {kl:.3f}")
        
        # Step 1: Probability ratio
        ratio = new_prob / old_prob
        print(f"\nStep 1: Probability ratio")
        print(f"  r = π_θ / π_old = {new_prob:.2f} / {old_prob:.2f} = {ratio:.3f}")
        
        # Step 2: Unclipped objective
        unclipped = ratio * advantage
        print(f"\nStep 2: Unclipped objective")
        print(f"  r * A = {ratio:.3f} * {advantage:.2f} = {unclipped:.3f}")
        
        # Step 3: Clipped objective
        clipped_ratio = max(1-epsilon, min(ratio, 1+epsilon))
        clipped = clipped_ratio * advantage
        print(f"\nStep 3: Clipped objective")
        print(f"  clip(r) = clip({ratio:.3f}, [{1-epsilon:.1f}, {1+epsilon:.1f}]) = {clipped_ratio:.3f}")
        print(f"  clip(r) * A = {clipped_ratio:.3f} * {advantage:.2f} = {clipped:.3f}")
        
        # Step 4: Final PPO objective
        final_obj = min(unclipped, clipped)
        pg_loss = -final_obj
        print(f"\nStep 4: PPO objective")
        print(f"  min(unclipped, clipped) = min({unclipped:.3f}, {clipped:.3f}) = {final_obj:.3f}")
        print(f"  PG Loss = -{final_obj:.3f} = {pg_loss:.3f}")
        
        # Step 5: Add KL penalty
        kl_penalty = beta * kl
        total_loss = pg_loss + kl_penalty
        print(f"\nStep 5: Add KL penalty")
        print(f"  KL penalty = β * KL = {beta} * {kl:.3f} = {kl_penalty:.4f}")
        print(f"  Total Loss = {pg_loss:.3f} + {kl_penalty:.4f} = {total_loss:.4f}")
        
        return total_loss


# Run all examples
if __name__ == "__main__":
    print("RLHF MATHEMATICAL EXAMPLES")
    print("="*60)
    
    # SFT example
    sft = SFTPhase(None)
    
    # Reward modeling example
    reward_modeling_example()
    
    # Probability ratio example
    probability_ratio_example()
    
    # Advantage example
    advantage_example()
    
    # Clip example
    clip_example()
    
    # Complete PPO example
    complete_ppo_example()
    
    # KL example
    kl_penalty_example()
    
    # Full PPO step
    ppo = PPOPolicy(None, None, None)
    ppo.numerical_example_step()
```

## 4. Complete RLHF Pipeline Summary

| Stage | Formula                                | What it does                           | Example Value |
| ----- | -------------------------------------- | -------------------------------------- | ------------- |
| SFT   | $-\sum \log P(y_t\|x,y_{<t})$          | Imitate good examples                  | Loss = 0.16   |
| RM    | $-\log \sigma(r_w - r_l)$              | Learn to predict preferences           | Loss = 0.27   |
| PPO   | $\min(rA, \text{clip}(r)A) - \beta KL$ | Optimize for reward while staying safe | Loss = -1.8   |

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br><span style="color:#E8600A;font-weight:700">RLHF</span> = <span style="color:#2980B9">SFT (模仿学习)</span> + <span style="color:#2980B9">Bradley-Terry (偏好建模)</span> + <span style="color:#2980B9">PPO with clipping & KL (安全优化)</span>. Each stage has a clear mathematical objective that translates directly to code.</div>
