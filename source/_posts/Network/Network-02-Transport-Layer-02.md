---
title: Transport-Layer-02
date: 2025-10-30 20:51
tags:
    - Network
    - Transport-Layer
math: true
syntax_converted: true

---


## 3.4 可靠数据传输 (Reliable Data Transfer - RDT) 原理

### RDT 总结

**目标**：在不可靠的信道 (Unreliable Channel) 上实现可靠数据传输。

**情况1：信道只会A出现bit错误 (Bit Errors)**
* **需求**：
    1.  接收端具有检查错误的能力，例如使用 **校验和 (Checksum)**。
    2.  接收端需要能
    够反馈接收状态，即 **确认 (Acknowledgement - ACK)**。
    3.  发送端在收到阴性确认 (Negative Acknowledgement - NAK) 或损坏的ACK时，需要有 **重传 (Retransmission)** 的能力。
* **新问题**：如果ACK消息也损坏或丢失，发送端可能会重传，导致接收端收到 **重复数据 (Duplicate Data)**。
* **解决方案**：引入 **序列号 (Sequence Number)**。接收端可以根据序列号丢弃重复的分组。
* **最终机制**：校验和 (Checksum)、确认 (ACK)、序列号 (Sequence Number)。

**情况2：信道既有bit错误，也会丢包 (Packet Loss)**
* **需求** (在情况1的基础上)：
    1.  发送端必须有能力检测到丢包。
* **解决方案**：设置 **定时器 (Timer)**。发送端在发送一个分组后启动定时器，如果定时器超时 (Timeout) 仍未收到ACK，就认为分组丢失，并重传。
* **新问题**：定时器超时并不一定意味着丢包，也可能是由于网络延迟过大。这同样会导致重复分组，但序列号机制已经解决了这个问题。
* **最终机制 (rdt3.0)**：校验和 (Checksum)、确认 (ACK)、序列号 (Sequence Number)、定时器 (Timer)。

### rdt3.0 的性能问题

* `rdt3.0` 协议功能正确，但性能极差。
* 它是一种 **停止-等待 (Stop-and-Wait)** 协议。发送端发送一个包后，必须停止并等待它的ACK，然后才能发送下一个包。
* **示例**：在一个 1 Gbps 的链路上，传播延迟 (Propagation Delay) 为 15ms，一个 8000 bit 的包：
    * 传输时间 $d_{trans} = L/R = 8000 \text{ bits} / 10^9 \text{ bps} = 8 \mu s$。
    * 往返时间 $RTT = 15 \text{ ms} \times 2 = 30 \text{ ms} = 30008 \mu s$ (计入传输时间)。
    * 发送端利用率 $U_{sender} = \frac{L/R}{RTT + L/R} = \frac{.008}{30.008} \approx 0.00027$。
* **结论**：发送端在绝大多数时间都在空闲等待ACK。网络协议严重限制了物理资源的使用。

### 流水线协议 (Pipelined Protocols)

* **解决方案**：允许发送方在收到ACK之前，连续发送多个分组。
* **优势**：极大地提高了信道利用率。
* **要求**：
    1.  需要更大的序列号范围。
    2.  发送方和/或接收方需要 **缓冲 (Buffering)**。
* **两种通用的流水线协议**：
    1.  **回退N步 (Go-Back-N - GBN)**
    2.  **选择重传 (Selective Repeat - SR)**

---

### 1. 回退N步 (Go-Back-N - GBN)

GBN 允许发送方有 $N$ 个已发送但未确认的分组在“飞行中”。

* **发送方 (Sender)**：
    * 维护一个大小为 $N$ 的 **发送窗口 (Send Window)**。
    * `send_base`：窗口中最早的未被确认的分组序列号。
    * `nextseqnum`：下一个待发送的分组序列号。
    * **累积确认 (Cumulative ACK)**：接收到 $ACK(n)$ 意味着序列号 $\leq n$ 的所有分组都已被正确接收。
    * **单一
    定时器 (Single Timer)**：只为最早的未确认分组 (`send_base`) 设置一个定时器。
    * **超时事件 (Timeout)**：如果定时器超时，发送方 **重传所有已发送但未确认的分组** (即从 `send_base` 到 `nextseqnum-1` 的所有分组)。
* **接收方 (Receiver)**：
    * **按序接收 (In-order)**：只维护一个 `expectedseqnum` (期望收到的序列号)。
    * 如果收到的分组序列号等于 `expectedseqnum`：接收并交付给上层，发送 $ACK(\text{expectedseqnum})$，然后 `expectedseqnum++`。
    * 如果收到 **乱序分组 (Out-of-order Packet)**：**直接丢弃 (Discard)**。
    * **无接收缓冲 (No Receiver Buffering)**：不缓冲任何乱序的分组。
    * **总是** 发送当前已正确接收的最高按序序列号的ACK。这会导致在丢包时，发送方会收到大量 **重复ACK (Duplicate ACKs)**。

### 2. 选择重传 (Selective Repeat - SR)

SR 通过让接收方缓冲乱序分组，来避免GBN中不必要的重传，只重传真正丢失的分组。

* **发送方 (Sender)**：
    * **为每个未确认的分组维护一个定时器 (Timer per Packet)**。
    * **超时事件 (Timeout)**：如果某个分组的定时器超时，**只重传那一个分组**。
    * **单独确认 (Individual ACK)**：接收来自接收方的对每个分组的单独ACK。
    * 发送窗口 (`sendbase`) 只有在窗口内的最早的分组被确认后才能向前滑动。
* **接收方 (Receiver)**：
    * **单独确认 (Individual ACK)**：对每个正确接收的分组（无论是否按序）都发送一个ACK。
    * **接收缓冲 (Receiver Buffering)**：**缓冲** 那些早于期望序列号（`rcv_base`）但未按序到达的分组。
    * **按序交付**：当 `rcv_base` 所指向的分组到达时，将它以及后续已缓冲的连续分组一起交付给上层，并向前移动接收窗口。
* **SR 的困境 (Dilemma)**：
    * 序列号空间的大小必须是窗口大小 ($N$) 的 **至少两倍**。
    * **原因**：如果序列号空间 $= N$（例如 N=3, 序列号 0,1,2），接收方无法区分一个序列号为 0 的分组是第一次发送的新包，还是一个迟到的重传包。

### GBN vs SR 总结

| 特性 (Comparison) | 回退N步 (Go-Back-N) | 选择重传 (Selective Repeat) |
| :--- | :--- | :--- |
| **重传 (Retransmission)** | 重传所有未确认的分组 | 只重传丢失/损坏的分组 |
| **ACK 类型** | 累积确认 (Cumulative) | 单独确认 (Individual) |
| **发送方定时器** | 1 个 (针对最早的未确认包) | N 个 (每个未确认包一个) |
| **接收方缓冲 (Buffering)** | 无需缓冲 (丢弃乱序包) | 必须缓冲乱序包 |
| **接收方排序 (Sorting)** | 无需排序 | 必须排序以按序交付 |
| **复杂度 (Complexity)** | 较简单 | 非常复杂 |
| **适用场景** | 网络条件好 (低丢包率) | 网络条件差 (高丢包率) |

---

## 3.5 面向连接的传输: TCP (Connection-Oriented Transport)

### TCP 概述 (Overview)

TCP 是互联网的核心传输层协议，它在IP协议（不可靠）之上提供可靠的数据传输服务。

* **点对点 (Point-to-point)**：一个发送方，一个接收方。
* **可靠的、按序的字节流 (Reliable, in-order byte stream)**：它不关心“报文”的边界，只是一个字节流。
* **流水线的 (Pipelined)**：通过 **拥塞控制 (Congestion Control)** 和 **流量控制 (Flow Control)** 来动态设置窗口大小。
* **全双工 (Full duplex)**：数据可以在同一连接上双向流动。这意味着<mark>数据可以同时在两个方向上流动</mark>（从客户端到服务器，_并且_ 从服务器到客户端）
* **面向连接 (Connection-oriented)**：在数据交换前，必须通过 **三次握手 (Three-Way Handshake)** 建立连接。
* **流量控制 (Flow controlled)**：确保发送方不会“淹没”（即超出缓冲区）接收方。

### 3.5.1 TCP 段结构 (Segment Structure)

TCP 在IP数据报中传输的数据单元称为 **段 (Segment)**。

<img src="Pasted image 20251030201030.png" alt="Pasted image 20251030201030">

* **源端口号 (Source Port #)** 和 **目的端口号 (Dest Port #)**：用于多路复用/分用。
* **序列号 (Sequence Number)**：
    * **关键**：TCP 序列号是 <mark>**字节流中的字节编号**</mark>，而不是段的编号。
    * 它指的是<mark>该段数据中 **第一个字节** 在整个字节流中的编号</mark>。
* **确认号 (Acknowledgement Number - ACK #)**：
    * **关键**：TCP 使用 **累积确认 (Cumulative ACK)**。
    * 这个字段的值是 <mark>**期望从对方接收的下一个字节的序列号**</mark>。
      <img src="Pasted image 20251030201141.png" alt="Pasted image 20251030201141">
* **首部长度 (Header Length)**：指示 TCP 首部的长度（因为有“选项”字段）。
* **标志位 (Flags)**：
    * `ACK`：表示“确认号”字段有效。
    * `SYN`：用于发起连接。
    * `FIN`：用于关闭连接。
    * `RST`：重置连接。
* **接收窗口 (Receive Window)**：用于 **流量控制**。该值告诉对方：“我的接收缓冲区还有多少字节的空闲空间”。
* **校验和 (Checksum)**：用于检查首部和数据的bit错误。

### 3.5.2TCP 可靠数据传输 (RDT)

TCP 的 RDT 机制结合了 GBN 和 SR 的思想，是一种高度优化的混合体。

#### 1. RTT 估计与超时 (RTT and Timeout)

TCP 必须设置一个超时定时器来重传丢失的分组，这个定时器的值至关重要。

* **问题**：RTT (Round Trip Time) 在网络中是动态变化的。
    * **定时器太短**：导致过早超时 (Premature Timeout)，进行不必要的重传，浪费带宽。
    * **定时器太长**：对丢包的反应慢，降低吞吐量。
* **解决方案**：动态估计 RTT。
    1.  **估计 RTT ($EstimatedRTT$)**：
        * 测量多个 $SampleRTT$（从发送段到收到ACK的时间）。
        * 使用 **指数加权移动平均 (EWMA)** 来平滑 $SampleRTT$ 的抖动。
        * $EstimatedRTT = (1-\alpha) \times EstimatedRTT + \alpha \times SampleRTT$ (典型 $\alpha = 0.125$)
    2.  **设置超时时间 ($TimeoutInterval$)**：
        * 还需要考虑 RTT 的 **方差 (Variance)**。
        * $DevRTT = (1-\beta) \times DevRTT + \beta \times |SampleRTT - EstimatedRTT|$ (典型 $\beta = 0.25$)
        * $TimeoutInterval = EstimatedRTT + 4 \times DevRTT$ (增加一个“安全边界”)

#### 2. TCP 重传机制 (Retransmission)

TCP 使用 **累积确认** 并维护一个 **单一的重传定时器**（逻辑上，它总是为最早的未确认段 `SendBase` 计时）。

**重传由两个事件触发：**
1.  **超时 (Timeout)**
2.  **重复的 ACK (Duplicate ACKs)**

* **简化的TCP发送方逻辑 (Simplified Sender)**：
    * **事件：收到来自应用的数据**：创建段，发送，如果定时器未运行，则启动定时器。
    * **事件：定时器超时**：**重传** 造成超时的段 (即 `SendBase` 对应的段)，重启定时器。
    * **事件：收到 ACK ($y$)**：如果 $y > SendBase$ (即它是新的确认)，则更新 $SendBase = y$。如果此时还有未确认的段，**重启定时器**。

* **丢包场景**：
  <img src="Pasted image 20251030201203.png" alt="Pasted image 20251030201203">
    * **ACK 丢失**：如果 `ACK=100` 丢失，发送方最终会因 `Seq=92` 的定时器超时而 **重传** `Seq=92`。接收方会丢弃这个重复的段，并 **再次发送** `ACK=100`。
    * **段丢失 (Segment Loss)**：如果 `Seq=92` 丢失，接收方将永远不会发送 `ACK=100` (因为ACK是累积的)。发送方最终会因 `Seq=92` 的定时器超时而 **重传** `Seq=92`。


| **接收方事件**                            | **TCP 接收方动作**                                   |
| ------------------------------------ | ----------------------------------------------- |
| 按序分段到达，具有期望的序列号。所有低于此期望序列号的数据均已被确认。  | 延迟 ACK。等待最多 500ms 以接收下一个分段。如果没有下一个分段到达，则发送 ACK。 |
| 按序分段到达，具有期望的序列号。另有一个分段的 ACK 正处于待定状态。 | 立即发送一个单一的累积 ACK，同时确认这两个按序分段。                    |
| 乱序分段到达，序列号高于期望值。检测到空隙。               | 立即发送_重复 ACK_，指明下一个期望字节的序列号。                     |
| 分段到达，该分段部分或完全填充了空隙。                  | 立即发送 ACK，前提是该分段从空隙的下边界开始。                       |

#### 3. 快速重传 (Fast Retransmit)

**问题**：等待超时重传的延迟太长。

**优化**：使用 **重复的ACK (Duplicate ACKs)** 来提早检测丢包。
* 当一个段丢失时（例如 `Seq=100` 丢失），但后续的段（`Seq=120`, `Seq=140`...）被接收方收到。
* 由于接收方期望的是 `Seq=100`，它会 **丢弃** 这些乱序的段（在经典TCP中）或者 **缓冲** 它们（在现代TCP中）。
* 但无论如何，它 **每次** 收到一个乱序段时，都会 **立即** 发送一个ACK，ACK的值是它 **期望的字节号**，即 `ACK=100`。
* 这导致发送方会收到多个 `ACK=100`，这就是 **重复ACK**。
* **快速重传规则**：当<mark>发送方收到 **3 个重复的ACK** (即总共4个相同的ACK) 时</mark>，它就认为这个ACK所指示的下一个段（`Seq=100`）已经丢失。
* **动作**：发送方 <mark>**立即重传 (Fast Retransmit)**</mark> 丢失的段，**而无需等待定时器超时**。

---

### 3.5.3 TCP 流量控制 (TCP Flow Control)

**目标**：防止发送方发送数据的速度超过 **接收方** 应用程序处理数据的速度，导致接收方缓冲区溢出。

* **区别**：这是为了保护 **接收方**，而 **拥塞控制** 是为了保护 **网络**。
* **机制**：
    1.  接收方在其接收缓冲区 `RcvBuffer` 中跟踪“空闲空间” `RcvWindow`。
    2.  $RcvWindow = RcvBuffer - (\text{LastByteRcvd} - \text{LastByteRead})$
    3.  接收方将这个 $RcvWindow$ 的值放入它发送的每个TCP段的 **“接收窗口” (Receive Window)** 字段中，通告给发送方。
    4.  发送方维护一个变量 `RcvWindow` (从ACK中获取)，并确保其“在途”（已发送但未确认）的数据量不超过该值。
    5.  **发送方限制**：$\text{LastByteSent} - \text{LastByteAcked} \leq RcvWindow$

---

### 3.5.4 TCP 连接管理 (TCP Connection Management)

#### 1. 建立连接：三次握手 (Three-Way Handshake)

* **Step 1: Client -> Server: `SYN`**
    * 客户端发送一个 `SYN` 段（`SYN=1`）。
    * 随机选择一个初始序列号 $x$（`seq=x`）。
* **Step 2: Server -> Client: `SYNACK`**
    * 服务器收到 `SYN` 后，分配缓冲区和变量。
    * 服务器发回一个 `SYNACK` 段（`SYN=1`, `ACK=1`）。
    * 随机选择自己的初始序列号 $y$（`seq=y`）。
    * 确认客户端的 `SYN`（`ack=x+1`）。
* **Step 3: Client -> Server: `ACK`**
    * 客户端收到 `SYNACK` 后，也分配缓冲区。
    * 客户端发送一个 `ACK` 段（`ACK=1`）。
    * 确认服务器的 `SYN`（`ack=y+1`）。
    * 此时 `SYN=0`。这个段可以携带应用层数据。

**思考：为什么是三次握手？**
* **不是两次？** 无法防止“已失效的连接请求”。如果一个旧的 `SYN` (来自 $x$) 在网络中延迟了很久才到达服务器，服务器会用 `SYNACK` (来自 $y$, $ack=x+1$) 回复并建立连接。客户端（早已超时）会忽略这个 `SYNACK`。但如果是两次握手，服务器会单方面建立连接并等待数据，浪费资源。

#### 2. 关闭连接：四次挥手 (Four-Way Handshake)

连接的两个方向必须被单独关闭。

* **Step 1: Client -> Server: `FIN`**
    * 客户端应用调用 `close()`。
    * 客户端发送一个 `FIN` 段（`FIN=1`, `seq=u`）。
* **Step 2: Server -> Client: `ACK`**
    * 服务器收到 `FIN`，发送一个 `ACK` 来确认（`ACK=1`, `ack=u+1`）。
    * 此时，客户端不能再发送数据，但服务器 **仍然可以发送数据**（进入 `CLOSE_WAIT` 状态）。
* **Step 3: Server -> Client: `FIN`**
    * 当服务器应用也调用 `close()`，并且它也发送完了所有数据时。
    * 服务器发送一个 `FIN` 段（`FIN=1`, `seq=v`）。
* **Step 4: Client -> Server: `ACK`**
    * 客户端收到 `FIN`，发送一个 `ACK` 来确认（`ACK=1`, `ack=v+1`）。

**TIME_WAIT 状态**
* 在第4步发送 `ACK` 后，**客户端** 会进入 `TIME_WAIT` 状态（通常等待 30 秒或 2*MSL）。
* **目的**：
    1.  为了可靠地终止连接。如果客户端的最后一个 `ACK` (第4步) 丢失，服务器会超时并重传 `FIN` (第3步)。`TIME_WAIT` 状态可以确保客户端能收到这个重传的 `FIN` 并重新发送 `ACK`。
    2.  为了防止“已失效的连接请求”中的延迟分组在新连接中被误解。

---

## 3.6 拥塞控制 (Congestion Control) 原理

### 什么是拥塞 (Congestion)？

* **非正式定义**：“太多的源端，发送了太多的数据，太快了，以至于<mark>网络</mark>无法处理”。
* **表现**：
    * **丢包 (Lost Packets)**：路由器的缓冲区溢出。
    * **长延迟 (Long Delays)**：分组在路由器的缓冲区中排队。

### 拥塞控制 vs 流量控制 (Congestion vs Flow Control)

这是一个非常重要的区别：
* **流量控制 (Flow Control)**：
    * **问题**：发送方太快，**接收方** 处理不过来。
    * **类比**：B 说：“你说慢点，我拿笔记一下”。
    * **解决**：TCP 使用 **接收窗口 (RcvWindow)** 字段。
* **拥塞控制 (Congestion Control)**：
    * **问题**：发送方（们）太快，**网络** 处理不过来。
    * **类比**：B 说：“你说慢点，我这信号不好（网络卡），听不清”。
    * **解决**：TCP 使用 **拥塞窗口 (CongWin)**。

### 拥塞的代价 (Costs of Congestion)

1.  **大延迟**：当输入速率接近链路容量时，排队延迟趋于无穷大。
2.  **不必要的重传**：由于延迟大导致过早超时，发送方重传了并未丢失的包，浪费网络带宽。
3.  **上游带宽浪费**：在一个多跳路径中，如果一个分组在下游链路被丢弃，那么它在所有上游链路中传输所占用的带宽就完全被浪费了。

### 拥塞控制的方法

1.  **端到端拥塞控制 (End-to-end Congestion Control)**：
    * 网络不提供明确的反馈。
    * 端系统通过观察 **丢包 (Loss)** 和 **延迟 (Delay)** 来 *推断* 网络是否拥塞。
    * **这是 TCP 采用的方法。**
2.  **网络辅助拥塞控制 (Network-assisted Congestion Control)**：
    * 路由器提供明确的反馈给端系统。
    * 例如，路由器可以设置一个 ECN (Explicit Congestion Notification) 位，或者直接告诉发送方一个明确的速率 (Explicit Rate)。

---

## 3.7 TCP 拥塞控制 (TCP Congestion Control)

#### **拥塞窗口 (Congestion Window)**


1. **功能**：拥塞窗口用于限制TCP发送方在任意时刻可以向网络中发送、但尚未收到确认的数据量 。发送方必须遵守规则：`LastByteSent - LastByteAcked \le CongWin` 。
2. **目的**：它反映了发送方对网络拥塞程度的感知 。它是一个动态变化的值 。
3. **速率控制**：发送方的传输速率（以字节/秒为单位）大致等于拥塞窗口的大小 (`CongWin`) 除以 RTT (往返时间) 。
#### AIMD (Additive Increase, Multiplicative Decrease) 是 TCP 拥塞控制的核心思想。

<img src="Pasted image 20251030201302.png" alt="Pasted image 20251030201302">

* **基本策略**：发送方不断增加其传输速率（通过增大拥塞窗口 `CongWin`）来探测可用带宽，直到发生丢包（检测到拥塞）。一旦发生丢包，就大幅削减其传输速率。
* **加法增大 (Additive Increase - AI)**：
    * 在“拥塞避免”阶段，`CongWin` 每经过一个 RTT (Round Trip Time) 就会增加 1 MSS (Maximum Segment Size)。
    * 这是一种线性的、缓慢的增长方式，用于温和地探测更多可用带宽。
* **乘法减小 (Multiplicative Decrease - MD)**：
    * 当检测到一次丢包事件（例如超时或收到3个重复ACK）时，发送方会将 `CongWin` 切割为一半。
    * 这是一种快速的、大幅度的速率降低，用于迅速缓解网络拥塞。
* **行为模式**：
    * 这种 AI 和 MD 结合的策略导致 `CongWin` 的变化呈现出一种“锯齿形” (Sawtooth) 行为。
    * 窗口大小线性增长，直到发生丢包，然后被迅速（乘法）减半，再开始新一轮的线性增长。
TCP 采用端到端的方法，通过 **丢包事件 (Loss Event)** 来推断拥塞。
* **丢包事件** = **超时 (Timeout)** 或 **收到3个重复ACK**。
* TCP 发送方维护一个 **拥塞窗口 (Congestion Window - `CongWin`)**。
* 发送方的实际发送速率 $\approx \text{CongWin} / \text{RTT}$。
* 发送方通过调整 `CongWin` 的大小来控制其向网络发送数据的速率。

TCP 拥塞控制由三个核心机制组成：
1.  **慢启动 (Slow Start - SS)**
2.  **拥塞避免 (Congestion Avoidance - CA)**
3.  **快速恢复 (Fast Recovery)**

#### 1. 慢启动 (Slow Start)

* **目的**：在连接刚开始时，快速地“探测”可用带宽，找到拥塞发生的“大致位置”。
* **机制**：
    * 连接开始时，设置 `CongWin = 1` MSS (Maximum Segment Size)。
    * <mark>**每当收到一个 ACK**</mark>，就将 `CongWin` 增加 1 MSS。
    * **效果**：这导致 `CongWin` **每经过一个 RTT 就翻倍**（$1 \to 2 \to 4 \to 8 \to 16...$）。
    * 这是一个 **指数增长 (Exponential Growth)** 过程。

#### 2. 拥塞避免 (Congestion Avoidance)

* **目的**：当 `CongWin` 增长到一定程度（即接近网络容量）时，转为一种更温和的探测方式，避免造成拥塞。
* **机制**：**AIMD**
    * **加法增大 (Additive Increase)**：**每经过一个 RTT**，`CongWin` 只增加 1 MSS。
    * **实现**：对于收到的 *每个* ACK，`CongWin += MSS * (MSS / CongWin)`。
    * 这是一个 **线性增长 (Linear Growth)** 过程。

* **SS 和 CA 的转换**：
    * TCP 维护一个 **阈值 (Threshold)** 变量。
    * 当 `CongWin < Threshold` 时：处于 **慢启动** 状态（指数增长）。
    * 当 `CongWin > Threshold` 时：处于 **拥塞避免** 状态（线性增长）。
    * 当 `CongWin == Threshold` 时：两者皆可。

#### 3. 快速恢复 (Fast Recovery)

* **目的**：对不同严重程度的丢包事件做出不同反应。
* **TCP Reno 算法**：
    * **事件1：超时 (Timeout)**
        * **认为**：这是一个 **严重** 的拥塞信号。
        * **动作**：
            1.  `Threshold = CongWin / 2`
            2.  **`CongWin = 1` MSS**
            3.  进入 **慢启动** 状态。
    * **事件2：收到3个重复ACK**
        * **认为**：这是一个 **轻微** 的拥塞信号（网络还能传包）。
        * **动作 (这就是快速恢复)**：
            1.  触发 **快速重传**（重传丢失的包）。
            2.  `Threshold = CongWin / 2`
            3.  **`CongWin = Threshold`** (乘法减小 - Multiplicative Decrease, MD)
            4.  进入 **拥塞避免** 状态 (而不是慢启动)。

* **TCP Tahoe 算法** (早期版本)：
    * 无论是超时还是 3 个重复 ACK，它都一律将 `CongWin` 设为 1 MSS 并进入慢启动。Reno 是一种优化。

<img src="Pasted image 20251030201320.png" alt="Pasted image 20251030201320">

<img src="Pasted image 20251030201335.png" alt="Pasted image 20251030201335">

- 处理时间是每次收到ACK都要进行处理，在收到ACK时候多发一个（增加一个MSS）

<img src="Pasted image 20251030201349.png" alt="Pasted image 20251030201349">

### TCP 拥塞控制的演进

* **基于丢包 (Loss-based)**：
    * **Tahoe**: 早期版本。
    * **Reno**: 经典版本，区分了超时和3-dup-ACK。
    * **Cubic**: 现代 Linux 内核的默认算法，为高带宽、低丢包率网络优化。
* **基于 RTT (RTT-based)**：
    * **Vegas**: 未被广泛采用。
* **基于链路容量 (Capacity-based)**：
    * **BBR (Bottleneck Bandwidth and Round-trip propagation time)**：由 Google 提出 (2016)，已在 Google、YouTube 部署，显著降低延迟。

### TCP 公平性 (Fairness)

* **目标**：如果 $K$ 个 TCP 连接共享一个带宽为 $R$ 的瓶颈链路，每个连接应平均获得 $R/K$ 的速率。
* **AIMD 实现了公平性**：两个竞争的连接会自然地收敛到平等的带宽共享。
* **不公平的来源**：
    1.  **UDP**：UDP 不使用拥塞控制。一个高速的 UDP 流会抢占 TCP 的带宽，导致 TCP 连接饿死（速率降到极低）。
    2.  **并行 TCP 连接 (Parallel TCP Connections)**：一个应用程序（如Web浏览器）可以同时打开多个（例如10个）TCP连接到同一个服务器。这10个连接会与其它应用程序的1个连接竞争，导致该应用获得 $\approx 10/(10+1)$ 的带宽，而其它应用只有 $1/(10+1)$。