---
title: Transport-Layer-01
date: 2025-10-30 20:49
tags:
    - Network
    - Transport-Layer
math: true
syntax_converted: true

---

## 一、传输层概述 (Transport Layer Services)

### 1.1 学习目标 (Goals)

* **理解传输层服务背后的原理 (Principles)**
    * 多路复用 (Multiplexing) / 多路分用 (Demultiplexing)
    * 可靠数据传输 (Reliable Data Transfer)
    * 流量控制 (Flow Control)
    * 拥塞控制 (Congestion Control)
* **学习互联网中的传输层协议 (Protocols)**
    * **UDP**: 无连接传输 (Connectionless Transport)
    * **TCP**: 面向连接的传输 (Connection-oriented Transport)
    * TCP 拥塞控制 (TCP Congestion Control)

### 1.2 传输层的核心功能

* 传输层协议为运行在不同主机上的<mark>**应用进程 (Application Processes)**</mark> 之间提供<mark>**逻辑通信 (Logical Communication)**</mark>。
* 传输协议只运行在**端系统 (End Systems)** 中。
    * **发送方 (Send Side)**: 将应用层的报文 (Messages) 分割成**报文段 (Segments)**，然后传递给网络层。
    * **接收方 (Rcv Side)**: 将接收到的报文段重组成报文，并传递给应用层。
* 互联网为应用提供了多种传输协议，主要是 TCP 和 UDP。

### 1.3 传输层 vs. 网络层 (Transport vs. Network Layer)

* **网络层 (Network Layer)**: 提供主机 (Hosts) 之间的逻辑通信。
* **传输层 (Transport Layer)**: 提供进程 (Processes) 之间的<font color="#ffc000">逻辑通信</font>。 它依赖于网络层提供的服务，并对其进行增强。

{% note default '**家庭类比 (Household Analogy):**' %}
* 12个孩子之间互相写信。
* 进程 (Processes) = 孩子们 (Kids)
* 应用报文 (App Messages) = 信封里的信件 (Letters)
* 主机 (Hosts) = 房子 (Houses)
* **传输层协议** = Ann 和 Bill（负责在屋里收发信件，确保信件交给正确的孩子）。
* **网络层协议** = 邮政服务 (Postal Service)（负责在房子之间传递信件）。

{% endnote %}
### 1.4 互联网协议栈中的数据单位

| 协议层                      | 数据基本单位                     |
| :-------------------------- | :------------------------------- |
| **应用层 (Application Layer)** | 消息 (Message)                 |
| **传输层 (Transport Layer)** | 数据段 (Segment)               |
| **网络层 (Network Layer)** | 数据包 (Datagram)              |
| **链路层 (Link Layer)** | 数据帧 (Frame)                 |
| **物理层 (Physical Layer)** | 符号 (Symbol)、比特 (Bit)      |

---

## 二、多路复用与多路分用 (Multiplexing and Demultiplexing)

* **多路分用 (Demultiplexing)**: 在接收方主机，将收到的报文段根据头部信息分发给正确的套接字 (Socket)。
* **多路复用 (Multiplexing)**: 在发送方主机，从多个套接字收集数据，并为每个数据块封装上头部信息（用于后续的分用），然后传递给网络层。

### 2.1 分用是如何工作的

主机使用 **IP 地址** 和 **端口号 (Port Numbers)** 将报文段定向到正确的套接字。
* 每个数据报 (Datagram) 都包含源 IP 地址和目的 IP 地址。
* 每个数据报携带一个传输层报文段 (Segment)。
* 每个报文段都包含源端口号 (Source Port #) 和目的端口号 (Dest Port #)。

### 2.2 无连接分用 (Connectionless Demultiplexing - UDP)

* 一个 UDP 套接字由一个包含<mark>**两个元素</mark>的元组 (Two-tuple)** 来标识：`(目的IP地址, 目的端口号)`。
* 当主机收到一个 UDP 报文段时，它检查报文段中的**目的端口号**，并将其定向到绑定该端口号的套接字。
* 因此，来自不同源 IP 地址或源端口号的 IP 数据报，只要它们的目的端口号相同，都会被定向到同一个 UDP 套接字。

### 2.3 面向连接的分用 (Connection-oriented Demultiplexing - TCP)

* 一个 TCP 套接字由一个包含**四个元素的元组 (Four-tuple)** 来标识：`(源IP地址, 源端口号, 目的IP地址, 目的端口号)`。
* 接收方主机会使用这**全部四个值**来将报文段定向到正确的套接字。
* 这意味着服务器可以为每一个连接的客户端维护一个独立的套接字，每个套接字都由其唯一的四元组标识。

---

## 三、无连接传输: UDP (User Datagram Protocol)

UDP 是一种 “精简” 或 “最基本” 的互联网传输协议。

### 3.1 UDP 的特点

* **尽力而为 (Best Effort)**: UDP 报文段可能会**丢失 (Lost)** 或**乱序 (Delivered out of order)** 到达。
* **无连接 (Connectionless)**: 在数据传输前，发送方和接收方之间没有**握手 (Handshaking)** 过程。 每个 UDP 报文段都是独立处理的。
* **为什么需要 UDP?**
    * **无连接建立延迟**: 省去了建立连接所需的时间。
    * **简单**: 发送方和接收方无需维护连接状态。
    * **头部开销小**: UDP 的头部只有 8 字节，非常小。
    * **无拥塞控制**: UDP 可以按照应用期望的速率发送数据，不会被拥塞控制限制。

### 3.2 UDP 的应用

* 常用于<mark>**流媒体</mark>应用 (Streaming Multimedia Apps)**，因为这类应用能容忍部分数据丢失 (Loss Tolerant)，但对传输速率敏感 (Rate Sensitive)。
* 其他用途包括 <mark>**DNS** 和 **SNMP**</mark>。
* 如果需要在 UDP 上实现可靠传输，必须在**应用层 (Application Layer)** 增加可靠性机制。

### 3.3 UDP 报文段格式与校验和 (Checksum)

* **UDP 头部**: 包括源端口、目的端口、长度和校验和。
* **校验和 (Checksum)**: 用于检测传输过程中报文段出现的**差错 (Errors)**，例如比特翻转。
    * **发送方**: 将报文段内容视为一系列16比特的整数，计算它们的**反码和 (1's Complement Sum)**（将进位的1加在最低位上），并将结果存入校验和字段。
    * **接收方**: 对收到的报文段重新计算校验和，并与头部中的校验和字段进行比对。 如果不匹配，则检测到错误。

---

## 四、可靠数据传输原理 (Principles of Reliable Data Transfer - RDT)

目标：在不可靠的信道上实现可靠的数据传输。 这是网络领域最重要的十大主题之一。 我们将通过有限状态机 (FSM) 逐步构建可靠数据传输协议。

### 4.1 RDT 1.0: 可靠信道上的可靠传输

* **假设**: 底层信道是完全可靠的，没有比特错误，也没有丢包。
* **实现**: 发送方从上层接收数据，打包后发送；接收方从信道接收数据，解包后交付给上层。

### 4.2 RDT 2.0: 应对**比特错误** (Channel with Bit Errors)

* **新情况**: 底层信道可能会翻转数据包中的比特，但是不会有丢包。<mark>接收方必须要有发现错误的机制</mark>
* **解决方案**: 引入**错误检测**和**接收方反馈**机制。
    * **校验和 (Checksum)**: 用于检测比特错误。
    * **确认 (Acknowledgements, ACKs)**: 接收方告知发送方数据包已成功接收，可以发送下一个数据包。
    * **否定确认 (Negative Acknowledgements, NAKs)**: 接收方告知发送方数据包已损坏。
    * **重传 (Retransmission)**: 发送方在收到 NAK 后，重传该数据包。
    
<img src="Chapter3_part1.pdf#page=37&rect=12,47,699,512" alt="Chapter3_part1, p.37">
### 4.3 RDT 2.1 & 2.2: 应对**损坏的ACK/NAK**

* **RDT 2.0 的致命缺陷**: 如果 ACK 或 NAK 本身在传输中损坏了怎么办？ 发送方无法确定接收方的状态，直接重传可能导致**重复数据包 (Duplicate Packets)**。
* **解决方案**:
    * **RDT 2.1**: 引入**序列号 (Sequence Number)**。 发送方为每个数据包添加序列号（如0和1交替）。 接收方通过检查序列号来判断是否是重复数据包，如果是则直接丢弃。
      <img src="Chapter3_part1.pdf#page=41&rect=27,64,678,499" alt="Chapter3_part1, p.41"><img src="Chapter3_part1.pdf#page=42&rect=5,39,711,507" alt="Chapter3_part1, p.42">
    * **RDT 2.2 (无 NAK 协议)**: 接收方只发送 ACK。 为了告知哪个包被正确接收，ACK 中必须包含被确认包的序列号。 如果发送方收到一个<mark>**重复的 ACK**，就执行与收到 NAK 相同</mark>的操作：重传当前数据包。
      <img src="Chapter3_part1.pdf#page=45&rect=6,8,711,524" alt="Chapter3_part1, p.45">

### 4.4 RDT 3.0: 应对**丢包** (Packet Loss)

* **新情况**: 底层信道不仅会出错，还可能丢失数据包（数据或ACK）。
* **解决方案**: 引入**定时器 (Countdown Timer)**。
    * 发送方在发送数据包后，启动一个定时器。
    * 如果在定时器超时前没有收到对应的 ACK，就认为数据包丢失，并**重传 (Retransmits)** 该数据包。
    * 序列号机制可以很好地处理因延迟而非丢失导致的重传（即重复包问题）。
* RDT 3.0 也被称为**停等协议 (Stop-and-Wait Protocol)**。
  <img src="Chapter3_part1.pdf#page=47&rect=20,53,681,510" alt="Chapter3_part1, p.47">

### 4.5 RDT 机制总结

为了在不可靠信道上实现可靠数据传输，最终的解决方案整合了四项核心机制：
1.  **校验和 (Checksum)**: 解决比特错误问题。
2.  **确认 (ACK)**: 提供接收方反馈。
3.  **序列号 (Sequence Number)**: 解决重复数据包问题。
4.  **定时器 (Timer)**: 解决丢包问题。

### 4.6 RDT 3.0 协议性能分析

 rdt3.0 协议虽然实现了可靠传输，但其“**停止-等待**”（Stop-and-Wait）机制在现代网络（高带宽、高延迟）中性能极差，导致带宽资源被严重浪费。

#### 1. 问题根源：停止-等待机制
- **工作模式**: 发送方每发送一个数据包，就必须**停止**并**等待**接收方返回确认（ACK）后，才能发送下一个。
- **时间浪费**: 在等待ACK返回的整个往返时间（RTT）内，发送方完全处于**空闲状态**，无法利用链路发送更多数据。
#### 2. 实例量化分析
PPT中的例子生动地展示了这个问题：
- **场景**: 在一个 1 Gbps 的高速链路上，往返延迟（RTT）为 30ms。
- **发送耗时 (`L/R`)**: 发送一个数据包本身只需要 **8微秒**。
- **等待耗时 (`RTT`)**: 而等待确认却需要 **30,000微秒**。
- **链路利用率 (`Utilization`)**: 计算得出，发送方真正在发送数据的时间仅占整个周期的 **0.027%**。超过 **99.9%** 的时间里，高速链路都处于闲置状态。

---

## 五、流水线协议 (Pipelined Protocols)

### 5.1 停等协议的性能问题

RDT 3.0 (停等协议) 虽然能工作，但性能极差。 发送方在大部分时间里都在等待 ACK，导致链路利用率 (Utilization) 非常低。

### 5.2 流水线 (Pipelining)

* **核心思想**: 允许发送方在收到 ACK 之前，连续发送多个数据包。 这就像流水线一样，让多个数据包同时处于“在途 (in-flight)”状态。
* **优点**: 极大地提高了链路利用率。
* **要求**:
    * 需要更大的序列号范围。
    * 发送方和/或接收方需要设置缓冲区 (Buffering)。

---

## 六、滑动窗口协议：回退N帧 (Go-Back-N) 与选择重传 (Selective Repeat)

流水线协议有两种通用形式：回退N帧 (Go-Back-N) 和选择重传 (Selective Repeat)。 它们都属于**滑动窗口 (Sliding Window)** 方法。

### 6.1 回退N帧 (Go-Back-N, GBN)

* **发送方**:
    * 维护一个大小为 N 的发送窗口，最多允许有 N 个未被确认的数据包在流水线中。
    * 只为**最老的、未被确认的**数据包设置一个计时器。
    * 当计时器超时，**重传所有已发送但未被确认的**数据包。
    * 当接收到窗口最左端的包的ACK时，**重置**计时器并**移动窗口**，发送移动后的窗口中未发送的数据包
* **接收方**:
    * 只发送**累积确认 (Cumulative ACKs)**。 `ACK(n)` 表示序列号为 n 及 n 之前的**所有数据包都已正确接收**。
    * 对于乱序到达的数据包，**直接丢弃 (Discard)**，不进行缓存。 接收方会重新发送具有最高有序序列号的 ACK。
<img src="Chapter3_part1.pdf#page=60&rect=19,46,710,517&color=yellow" alt="Chapter3_part1, p.60">
<img src="Chapter3_part1.pdf#page=63&rect=37,42,696,510&color=yellow" alt="Chapter3_part1, p.63">

### 6.2 选择重传 (Selective Repeat, SR)
* **发送方**:
    * 同样维护一个大小为 N 的发送窗口。
    * 为**每一个未被确认的**数据包都维护一个<mark>独立</mark>的计时器。
    * 当某个计时器超时，**只重传那一个未被确认的**数据包。
* **接收方**:
    * 对**每一个正确接收的**数据包都进行单独确认 (Individual ACKs)。
    * 对于乱序到达的数据包，会将其**缓存 (Buffers)** 起来，直到所有缺失的数据包都到达，再按序交付给上层。
    * 当收到在窗口前的数据包，可能因为之前发送的ACK损坏导致sender计时器超时，这时候再次发送ACK(n)。
* 发送方和接收方窗口不对齐，是因为sender尚未收到ACK
<img src="Chapter3_part1.pdf#page=65&rect=19,37,709,498&color=yellow" alt="Chapter3_part1, p.65">
### 6.3 GBN 与 SR 对比总结

| 对比项         | 回退N帧 (Go-Back-N)                  | 选择重传 (Selective Repeat)          |
| :------------- | :----------------------------------- | :----------------------------------- |
| **重传机制** | 重传所有已发送但未确认的帧。         | 只重传丢失或损坏的帧。               |
| **接收方行为** | 丢弃所有乱序的帧。                   | 缓存乱序的帧，直到空缺被填补。       |
| **确认机制** | 累积确认 (Cumulative ACK)。          | 单独确认 (Individual ACK)。           |
| **复杂度** | 比较简单。                           | 逻辑更复杂，发送和接收方都需要排序和存储。 |
| **适用场景** | 适用于网络条件好、误码率低的情况。   | 适用于网络条件差、误码率高的情况。   |