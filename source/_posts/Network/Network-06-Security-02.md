---
title: Network - 06.Security-02
date: 2026-01-15 11:40:00 +0800
tags:
    - Network
categories:
    - Network
math: true
syntax_converted: true
---


# Chapter 8: Network Security (Part 2)

## 1. Network Security Attack Events (网络安全攻击事件案例)

### 1.1 Stuxnet ("震网"病毒)
* **性质**: 世界上首个网络“超级破坏性武器”，专门攻击工业控制系统 (DCS, SCADA, PLC) 的蠕虫病毒。
* **目标**: 2010年攻击伊朗核设施（Natanz 铀浓缩工厂）。
* **后果**:
    * 摧毁了 984 台铀浓缩离心机 (Centrifuges)。
    * 感染了 20 多万台计算机。
    * 导致 1000 台机器物理退化，伊朗核计划倒退两年。

### 1.2 BlackEnergy (乌克兰电网攻击)
* **性质**: 具有信息战水准的网络攻击事件，主要通过木马病毒和 DDoS 插件。
* **过程**:
    * 最初感染 `.exe` 文件 -> 释放代码 -> 创建解密驱动 -> 感染 `svchost.exe` -> 注入 DLL -> 下载 DDoS 插件。
* **后果**:
    * 2015年乌克兰电网遭到 BlackEnergy 病毒的 DDoS 攻击。
    * 导致 23 万人断电，断电时间长达 6 小时。
    * 这是首个成功攻击电网的恶意程序。

### 1.3 Colonial Pipeline Attack (Darkside 黑客团体)
* **性质**: 针对美国能源基础设施最具破坏性的网络攻击（Ransomware/勒索软件）。
* **事件**: 2021年，Darkside 黑客团体勒索科洛尼尔管道公司 (Colonial Pipeline)。
* **后果**:
    * 美国 18 个州因输油管停用陷入紧急状态（Gas Outages）。
    * 公司最终支付 440 万美元赎金。

---

## 2. Securing TCP Connections: SSL/TLS

### 2.1 概述与历史
* **SSL (Secure Sockets Layer)** / **TLS (Transport Layer Security)** 是广泛部署的安全协议。
* **支持**: 几乎所有的浏览器和 Web 服务器 (HTTPS)。HTTP与HTTPS的核心区别在于是否使用 **SSL/TLS** 协议进行加密和安全保护。
* **目标**: 主要是为 Web 电子商务交易设计（加密信用卡号等），提供机密性 (Confidentiality)、完整性 (Integrity) 和认证 (Authentication)。
* **位置**: 位于<mark>应用层 (Application) 和传输层 (TCP) </mark>之间。SSL 为应用程序提供 API（通常是 C 或 Java SSL 库）。
* **历史演变**:
    * 1993: Netscape 设计 SSL 1.0 (未发布)。
    * 1995: SSL 2.0 发布 (有严重漏洞)。
    * 1996: SSL 3.0 问世，大规模应用。
    * 1999: ISOC 发布 TLS 1.0 (SSL 的升级版)。
    * 2006/2008/2011: TLS 1.1, 1.2 及其修订版。

### 2.2 Toy SSL: 一个简单的安全通道模型

为了理解 SSL，课件首先构建了一个简化的模型 ("Toy SSL")，包含四个阶段：

1.  **Handshake (握手)**: Alice 和 Bob 使用证书*certificate*和私钥*private key*进行身份认证，并交换共享密钥 (Shared Secret)。
	1. **打招呼与亮身份**：
		- Alice 先发一个“Hello”打招呼 。
		- Bob 回复他的**证书 (Certificate)** 。证书就像身份证，里面包含了 Bob 的**公钥 ($K_B^+$​)**。Alice 通过验证证书，就能确定对方真的是 Bob，而不是冒充者 。
	2. **交换秘密 (Master Secret)**：
		- Alice 生成一个随机的数，称为**主密钥 (Master Secret, MS)** 。
		- 为了不让别人偷看，Alice 使用 Bob 的公钥加密这个 MS，生成加密后的主密钥 ($EMS$)，发送给 Bob 。
		- **原理**：只有拥有私钥 ($K_B^-$) 的 Bob 才能解开这个包裹拿到 MS。这样，双方就安全地共享了一个秘密 。
2.  **Key Derivation (密钥派生)**:
    * 虽然双方都有了主密钥 (MS)，但直接用这一把钥匙做所有事情（加密、校验、发数据、收数据）是不安全的。如果这把钥匙泄露，所有安全防线都会崩塌。
    * **原则**：密码学中有一个重要原则，不同的操作应该使用不同的密钥 。
    * 使用 Key Derivation Function (KDF) 从 Master Secret 派生出 4 个密钥：
        * $K_c, K_s$: 客户端/服务端加密密钥 (Encryption Keys)。
        * $M_c, M_s$: 客户端/服务端 MAC 密钥 (MAC Keys)。
    * **意义**：将加密和完整性校验分开，也将发送和接收的方向分开，大大提高了安全性
3.  **Data Transfer (数据传输)**:
	- TCP 协议传输的是字节流，但为了安全校验，我们需要把数据切成块。数据流被分割成一系列的 **Records (记录)**。每个 Record 包含 MAC 用于完整性检查。
    * **Sequence Numbers (序列号)**: 为了防止重放攻击 (Replay) 或重新排序，MAC 计算中包含序列号：$MAC = MAC(Mx, sequence || data)$。注意：序列号不直接在网络上传输，而是隐含的。
4.  **Connection Closure (连接关闭)**:
	- 普通的 TCP 连接断开（FIN 报文）是不加密的，这给攻击者留下了可乘之机。
    * 防止截断攻击 (Truncation Attack，即攻击者伪造 TCP FIN)。
    * 使用 Record Types (记录类型): Type 0 表示数据，Type 1 表示关闭连接。

### 2.3 Real SSL (真实的 SSL 协议)

#### SSL Cipher Suite (密码套件)

客户端和服务端必须协商决定使用哪组加密算法：
* **Public-key algorithm**: 公钥算法 (如 RSA)。
* **Symmetric encryption algorithm**: 对称加密算法 (如 DES, 3DES, RC2, RC4)。
* **MAC algorithm**: 消息认证码算法。

#### SSL Handshake Protocol (握手协议步骤)

1.  **Client Hello**: 发送支持的算法列表 + Client Nonce (随机数)。
2.  **Server Hello**: 选择算法 + 发送 Server Certificate + Server Nonce。
3.  **Client Key Exchange**: 验证证书，提取公钥，生成 `Pre_Master_Secret`，用服务器公钥加密后发送。
4.  **Key Generation**: 双方利用 `Nonces` 和 `Pre_Master_Secret` 独立计算出 Master Secret，进而生成加密密钥和 MAC 密钥。
5.  **Finished Messages**:
    * Client 发送所有握手消息的 MAC。
    * Server 发送所有握手消息的 MAC。
    * *目的*: 防止中间人攻击 (Man-in-the-middle) 篡改握手过程（如删除强加密算法）。
    * **结果**： 由于双方计算 MAC 的输入数据不一致，计算出的 MAC 值自然不同。Server 会发现 Client 发来的 MAC 与自己算的不匹配，从而意识到握手过程被篡改，立即断开连接 。

*注意*: 握手中使用 **Nonces** (随机数) 是为了防止重放攻击 (Replay Attack)。即使攻击者重放昨天的加密记录，由于 Nonce 改变，密钥也会改变，解密将失败。

#### SSL Record Protocol (记录协议)

* 结构: Header (Content type, Version, Length) + Data + MAC。
* 数据和 MAC 会被加密 (Symmetric algorithm)。
* Fragment (分片) 大小通常为 $2^{14}$ bytes (~16KB)。

---
### 3. Network Layer Security: IPsec

IPsec (IP Security) 是一套协议簇，旨在为网络层（IP层）通信提供安全性。由于所有互联网通信（如 TCP、UDP、ICMP）最终都要封装在 IP 数据包中传输，因此保护了 IP 层就相当于为整个网络提供了 **“全面覆盖 (Blanket Coverage)”** 的安全保护。

#### 3.1 概述与 VPN (Virtual Private Networks)

* **设计目标**：
    * IPsec 的核心在于确保 IP 数据报 (Datagrams) 的**机密性 (Confidentiality)**、**完整性 (Integrity)** 和 **源认证 (Origin Authentication)**。
    * 它可以加密数据报的 Payload（载荷），这意味着上层的 Web 浏览、E-mail、文件传输等都会自动被加密，应用程序无需修改即可受益。

* **VPN (虚拟专用网络) 的工作机制**:
    * **背景**: 许多机构（如公司总部与分支机构）需要安全的私有网络通信。传统的做法是租用昂贵的物理专线。
    * **VPN 方案**: 利用公共的、不安全的 Internet 来传输私有数据。
    * **“隧道”概念**: 当位于分公司的员工发送数据给总部时，分公司的网关路由器会将数据加密，并通过 Internet 发送给总部的网关路由器。
    * **过程**:
        1.  **加密**: 数据离开内部网络进入公共 Internet 前，被网关加密。
        2.  **传输**: 加密后的数据像普通 IP 包一样在 Internet 上路由（虽然黑客可以截获，但无法解密）。
        3.  **解密**: 到达目的地网关后，数据被解密并转发给内部的主机。
    * **优势**: 成本极低（使用现有的 Internet 接入），灵活性高（员工在酒店、家中也能通过 VPN 软件接入）。

#### 3.2 IPsec 的两种运行模式 (Modes)

IPsec 可以在两种不同的模式下运行，区别主要在于**IPsec 头部插入的位置**以及**保护的范围**。

**1. Transport Mode (传输模式)**
* **保护范围**: 仅保护数据包的 **Payload (载荷)**，即上层协议数据（如 TCP 段、UDP 段、ICMP 消息）。**原 IP 头部不被加密**。
* **报文结构**:
    * `[原 IP 头部] + [IPsec 头部] + [加密的 Payload]`
* **应用场景**:
    * 主要用于**端系统到端系统 (End-to-End)** 的通信。
    * 例如：两台服务器之间直接进行加密通信。
    * 缺点：路由器仍能看到源和目的 IP，流量分析攻击可能依然有效。

**2. Tunnel Mode (隧道模式) —— 最常用**
* **保护范围**: 保护**整个原 IP 数据包**（包括原 IP 头部和载荷）。
* **报文结构**:
    * IPsec 将原来的整个 IP 包当作“数据”，对其加密，然后封装在一个**新的 IP 数据包**中。
    * `[新 IP 头部] + [IPsec 头部] + [加密的原 IP 头部 + 原 Payload]`
* **应用场景**:
    * 主要用于**网关到网关 (Gateway-to-Gateway)** 或 **主机到网关** 的通信（即 VPN）。
    * **优势**: 真正的源 IP 和目的 IP 被隐藏在加密载荷中，外部网络（Internet）只能看到新 IP 头部（通常是两个网关的地址），从而隐藏了内部网络拓扑结构。端系统（如公司内部电脑）不需要安装 IPsec 软件，由网关透明处理。

#### 3.3 IPsec 的两个核心协议

IPsec 定义了两个协议来提供不同的安全服务，它们可以单独使用，也可以组合使用。

**1. AH (Authentication Header) - 认证头部协议**
* **提供的服务**:
    * **源认证 (Source Authentication)**: 确认发送者身份。
    * **数据完整性 (Data Integrity)**: 确保数据在传输中未被篡改。
    * **防重放攻击 (Anti-Replay)**。
* **局限性**: AH **不提供机密性 (Confidentiality)**。也就是说，AH 不会对数据进行加密，数据依然是明文传输的，只是你无法篡改它。
* **现状**: 由于缺乏加密功能，现代安全需求中 AH 使用较少。

**2. ESP (Encapsulation Security Payload) - 封装安全载荷协议**
* **提供的服务**:
    * 包含 AH 的所有功能（源认证、数据完整性、防重放）。
    * **关键特性**: 提供**机密性 (Confidentiality)**，即数据加密。
* **地位**: ESP 是目前应用最广泛的协议。

#### 总结：最佳实践组合

**Tunnel Mode with ESP (隧道模式 + ESP)** 是构建 VPN 的标准配置：
* **ESP** 保证了数据既不会被篡改，也不会被窃听。
* **Tunnel Mode** 允许在两个安全网关之间建立虚拟通道，隐藏了内部网络的 IP 结构，非常适合跨越公共 Internet 连接两个私有网络。
---

## 4. Operational Security: Firewalls and IDS

### 4.1 Firewalls (防火墙)

防火墙是网络安全的第一道防线，它将组织内部的“可信网络”与外部公共的“不可信 Internet”进行隔离。

* **核心功能**: 通过允许某些分组通过并阻止其他分组，来控制进出网络的流量。
* **主要防御目标 (Why Firewalls?)**:
    1.  **拒绝服务攻击 (DoS)**: 防止攻击者通过建立大量伪造的 TCP 连接（如 SYN flooding）耗尽服务器资源，导致正常用户无法访问。
    2.  **非法访问与篡改**: 防止攻击者修改或窃取内部敏感数据（例如，攻击者替换 CIA 主页）。
    3.  **访问控制**: 仅允许经过身份认证的用户或主机访问内部网络。

#### 三种防火墙类型详解

**1. Stateless Packet Filtering (无状态包过滤)**

这是最基础的防火墙形式，通常集成在路由器中。
* **工作机制**: 路由器独立处理每一个到达的数据包 (Packet-by-packet)，不保留任何历史记录或上下文信息。
* **决策依据**: 仅根据 IP/TCP/UDP 头部字段进行判断:
    * 源 IP 地址 / 目的 IP 地址。
    * 源端口号 / 目的端口号。
    * 协议类型 (TCP, UDP, ICMP)。
    * TCP 标志位 (SYN, ACK 等)。
* **访问控制列表 (ACL)**: 管理员配置一组规则表，防火墙**自上而下**匹配。一旦匹配成功，即执行对应动作 (Allow/Deny)。
    * *策略示例*:
        * **禁止外部 Web 访问**: 丢弃所有目的端口为 80 的出站数据包。
        * **仅允许特定连接**: 丢弃所有进来的 TCP SYN 包（阻止外部发起连接），但允许特定 Web 服务器除外。
        * **防 Smurf 攻击**: 丢弃所有发往广播地址的 ICMP 包。

**2. Stateful Packet Filtering (有状态包过滤)**

无状态过滤有时过于“粗暴”且不智能（例如，它无法区分一个 ACK 包是回应现有连接的，还是攻击者伪造的）。有状态过滤解决了这个问题。
* **工作机制**: 防火墙维护一张 **连接状态表 (Connection State Table)**，跟踪每个 TCP 连接的生命周期。
    * 跟踪连接建立 (SYN) 和 拆除 (FIN)。
    * 设定超时机制，自动清理非活动连接。
* **决策逻辑**:
    * 不仅检查头部字段，还检查该数据包是否属于当前已存在的合法连接。
    * **示例**: 如果一个进来的数据包标志位是 ACK（声称是响应），但状态表中没有对应的连接记录，防火墙会判定其“毫无意义 (make no sense)”并丢弃。
* **优势**: 比无状态过滤更安全，能过滤掉伪造的后续数据包。

**3. Application Gateways (应用网关)**

上述两种过滤只看网络层和传输层头部，应用网关则深入到**应用层**。
* **工作机制**: 针对特定的应用（如 Telnet, FTP, HTTP）设置专门的代理服务器。
* **流程示例 (Telnet)**:
    1.  内部用户不能直接 Telnet 到外部主机，路由器会拦截并阻断。
    2.  用户必须先 Telnet 到应用网关。
    3.  网关进行用户身份认证。
    4.  认证通过后，**网关**代理用户与外部主机建立连接，并在两个连接之间转发数据。
* **优势**: 可以深入检查应用层数据内容，实现基于用户身份的精细控制。

---

#### 防火墙的局限性 (Limitations)

尽管防火墙很有用，但并非万能：
1.  **IP Spoofing (IP 欺骗)**: 路由器只能看 IP 头部，无法验证该 IP 是否真的来自声称的源头，攻击者可以伪造源 IP。
2.  **可用性与复杂性**: 如果要支持多种应用，需要为每个应用部署特定的网关（或复杂的代理配置）。
3.  **客户端配置**: 客户端软件（如浏览器）通常需要手动配置代理地址才能使用网关。
4.  **UDP 处理困难**: 由于 UDP 无连接，防火墙很难判断状态，往往采取“全部允许”或“全部拒绝”的策略，缺乏灵活性。
5.  **内部威胁**: 防火墙防外不防内，无法阻止内部员工的恶意攻击或误操作。
6.  **加密流量**: 如果攻击流量被封装在加密隧道（如 SSL 或 VPN）中，防火墙无法检查内容。

---

### 4.2 Intrusion Detection Systems (IDS, 入侵检测系统)

防火墙通常只检查头部 (Packet Filtering)，而 IDS 则进行更深度的分析，用于检测已穿透防火墙的攻击或内部威胁。

* **核心技术**:
    1.  **Deep Packet Inspection (DPI, 深度包检测)**:
        * IDS 不仅看头部，还深入检查数据包的**载荷内容 (Contents)**。
        * **特征匹配**: 将包内的字符串与已知的病毒特征库、攻击代码数据库进行比对。
    2.  **Correlation (关联分析)**:
        * 不只看单个包，而是分析**多个数据包**或**多个会话**之间的关联。
        * **应用场景**:
            * **Port Scanning (端口扫描)**: 检测某个源 IP 是否在短时间内尝试连接大量不同的端口。
            * **Network Mapping (网络测绘)**: 检测探测网络拓扑的行为。
            * **DoS Attack**: 检测流量异常峰值。

* **部署架构**:
    * **Sensors (传感器)**: 分布在网络的不同关键节点。
        * 例如：防火墙外部、防火墙内部、**DMZ (非军事化区)** 区域（放置 Web/FTP/DNS 服务器的地方）。
    * 多点部署允许 IDS 对比不同位置的流量，从而精确定位攻击源和受影响范围。
---

## 5. 思考题：智慧家庭安全协议设计 (Thinking Exercise)

假设为智慧家庭设计通信协议，涉及：IoT 设备 ($ID_1...ID_n$)、网关 ($ID_0$)、数据云、APP。

**需求**: 数据安全、保密性、身份认证、抗重放攻击、轻量级。

### 设计思路参考
1.  **密钥分配**:
    * 每个设备配置唯一密钥 $k_i$。
    * 网关 $ID_0$ 存储所有设备的密钥 $k_i$。
    * 网关拥有自己的密钥 $k_0$，存储在云端。

2.  **设备到网关的认证 (Lightweight)**:
    * 数据格式: $ID || E_{ki}(ID || Data)$
    * 网关收到后，根据明文 ID 查找 $k_i$ 进行解密。
    * 比对解密出的 ID 与明文 ID 是否一致（认证 + 完整性）。

3.  **网关到云端的数据上传**:
    * 汇聚数据: $DATA = (ID_1 || Data_1) + (ID_2 || Data_2)...$
    * 使用 $E_{k0}$ 加密传输。
    * 引入 Timestamp 或 Nonce 防止重放。