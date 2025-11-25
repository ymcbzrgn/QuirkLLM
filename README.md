# 🚀 QuirkLLM

## Lokal, Ücretsiz, Akıllı Kodlama Asistanı
### Claude Code CLI & Gemini CLI'ın GPU'suz Alternatifi

---

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  $ quirkllm                                                             │
│                                                                         │
│  ╭─────────────────────────────────────────────────────────────────╮   │
│  │  🚀 QuirkLLM v1.0.0 - Interactive Coding Assistant              │   │
│  │  📁 Project: ~/my-react-app (React 18, TypeScript)              │   │
│  │  🧠 Model: QuirkLLM-1.3B (4-bit) | Context: 32K                 │   │
│  │  💾 RAM: 16GB detected → Comfort Mode                           │   │
│  │  📡 Mode: Online (RAG active)                                   │   │
│  ╰─────────────────────────────────────────────────────────────────╯   │
│                                                                         │
│  > create a user profile component with avatar and bio                  │
│                                                                         │
│  🤖 Creating component...                                               │
│                                                                         │
│  ✓ Created: src/components/UserProfile.tsx                              │
│  ✓ Created: src/components/UserProfile.css                              │
│                                                                         │
│  > now add a hover animation with framer motion                         │
│                                                                         │
│  🤖 I'll add Framer Motion animation to the component I just created.  │
│                                                                         │
│  ✓ Updated: src/components/UserProfile.tsx                              │
│  ✓ Running: yarn add framer-motion                                      │
│                                                                         │
│  > /status                                                              │
│                                                                         │
│  ╭─ System Status ──────────────────────────────────────────────────╮  │
│  │  RAM: 4.2GB / 16GB (26%)                                         │  │
│  │  Context: 8,421 / 32,768 tokens (25%)                            │  │
│  │  Session: 12 messages (auto-save on)                             │  │
│  │  Cache Hit Rate: 73%                                             │  │
│  ╰──────────────────────────────────────────────────────────────────╯  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 📋 İÇİNDEKİLER

1. [Nedir?](#-nedir)
2. [RAM-Aware Adaptive System](#-ram-aware-adaptive-system)
3. [4 Çalışma Modu](#-4-çalışma-modu-shifttab)
4. [Neden QuirkLLM?](#-neden-quirkllm)
5. [Özellikler](#-özellikler)
6. [Kullanım Örnekleri](#-kullanım-örnekleri)
7. [Teknik Detaylar](#-teknik-detaylar)
8. [Mimari](#-mimari)
9. [78 Bileşen](#-78-bileşen)
10. [CLI Komutları](#-cli-komutları)
11. [Context Yönetimi](#-context-yönetimi)
12. [Fine-Tuning](#-fine-tuning-axolotl)
13. [RAG Sistemi](#-rag-sistemi)
14. [Test Suite](#-test-suite)
15. [Roadmap](#-roadmap)
16. [Kurulum](#-kurulum)
17. [Kararlar](#-kararlar)
18. [Lisans](#-lisans)

---

# 🎯 NEDİR?

QuirkLLM, **Claude Code CLI** ve **Gemini CLI** gibi çalışan, ancak:
- 💰 **Tamamen ücretsiz**
- 🏠 **%100 lokal**
- � **GPU gerektirmez** (Varsa otomatik kullanır ve hızlanır)
- 🔒 **Gizlilik odaklı**
- 🧠 **RAM-Aware** - Sisteminize göre otomatik optimize

bir **interaktif kodlama asistanıdır**.

## Tek Satırda

```bash
$ quirkllm   # Claude Code CLI gibi, ama lokal ve ücretsiz
```

## Fark Yaratan Özellik: RAM-Aware Adaptive System

```
8GB RAM?   → Survival Mode   (16K context, 4-bit, temel özellikler)
16GB RAM?  → Comfort Mode    (32K context, 4-bit, tüm özellikler)
32GB RAM?  → Power Mode      (64K context, 8-bit, paralel işlem)
64GB+ RAM? → Beast Mode      (128K context, 8-bit, maksimum her şey)
```

**Sistem otomatik algılar, siz sadece `quirkllm` yazın!**

---

# 🧠 RAM-AWARE ADAPTIVE SYSTEM

## Otomatik Profil Seçimi

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 RAM-AWARE ADAPTIVE SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  $ quirkllm                                                             │
│                                                                         │
│  🔍 Detecting system resources...                                       │
│  💾 RAM: 32GB Total | 14GB Available (used for profile)                 │
│  ⚡ Profile: COMFORT MODE activated                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              COMFORT MODE (12GB+ Available)                     │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Context Window  : 32,768 tokens                                │   │
│  │  Quantization    : 4-bit (balanced)                             │   │
│  │  Batch Size      : 4                                            │   │
│  │  RAG Cache       : 500MB                                        │   │
│  │  KV Cache        : 4GB                                          │   │
│  │  Concurrent Ops  : 2                                            │   │
│  │  Features        : All enabled                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4 RAM Profili (Boşta Olan RAM'e Göre)

### 🟡 SURVIVAL MODE (< 8GB Available)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SURVIVAL MODE - < 8GB FREE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  "Sistem dolu, dikkatli ilerliyoruz"                                   │
│                                                                         │
│  Ayarlar:                                                               │
│  ────────                                                               │
│  Context Window    : 16,384 tokens (16K)                               │
│  Quantization      : 4-bit (aggressive)                                │
│  Batch Size        : 1 (sequential)                                    │
│  RAG Cache         : 200MB                                             │
│  KV Cache          : 2GB                                               │
│  Embedding         : CodeBERT-small (250MB)                            │
│  Concurrent Ops    : 1 (single thread)                                 │
│  Model Loading     : Lazy (layer by layer)                             │
│  Compaction        : Aggressive (son 3 mesaj tam)                      │
│                                                                         │
│  Bellek Dağılımı:                                                      │
│  ─────────────────                                                     │
│  Model (4-bit)     ████████████░░░░░░░░░  1.5GB                        │
│  CodeBERT-small    ██░░░░░░░░░░░░░░░░░░░  0.25GB                       │
│  KV Cache          ████████░░░░░░░░░░░░░  2.0GB                        │
│  RAG Cache         █░░░░░░░░░░░░░░░░░░░░  0.2GB                        │
│  Context Buffer    ██░░░░░░░░░░░░░░░░░░░  0.5GB                        │
│  System            ██░░░░░░░░░░░░░░░░░░░  0.5GB                        │
│  Buffer (safety)   ███████░░░░░░░░░░░░░░  3.0GB                        │
│  ───────────────────────────────────────────────                       │
│  TOPLAM            ████████████████████░  ~8GB                         │
│                                                                         │
│  Özellik Durumu:                                                       │
│  ─────────────────                                                     │
│  ✅ Interactive Chat                                                   │
│  ✅ Multi-turn Conversation                                            │
│  ✅ Context Memory                                                     │
│  ✅ Basic RAG                                                          │
│  ✅ File Operations                                                    │
│  ⚠️  Semantic Cache (limited)                                          │
│  ⚠️  Parallel Search (disabled)                                        │
│  ⚠️  Large File Analysis (chunked)                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🟢 COMFORT MODE (8GB - 24GB Available) - ÖNERİLEN

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   COMFORT MODE - 16GB FREE ⭐                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  "İdeal deneyim, hiçbir kısıtlama hissetmezsin"                        │
│                                                                         │
│  Ayarlar:                                                               │
│  ────────                                                               │
│  Context Window    : 32,768 tokens (32K)                               │
│  Quantization      : 4-bit (balanced)                                  │
│  Batch Size        : 4                                                 │
│  RAG Cache         : 500MB                                             │
│  KV Cache          : 4GB                                               │
│  Embedding         : CodeBERT-base (500MB)                             │
│  Concurrent Ops    : 2                                                 │
│  Model Loading     : Hybrid (critical layers eager)                    │
│  Compaction        : Smart (son 5 mesaj tam)                           │
│                                                                         │
│  Bellek Dağılımı:                                                      │
│  ─────────────────                                                     │
│  Model (4-bit)     ████████░░░░░░░░░░░░░  1.5GB                        │
│  CodeBERT-base     ███░░░░░░░░░░░░░░░░░░  0.5GB                        │
│  KV Cache          █████████████████░░░░  4.0GB                        │
│  RAG Cache         ███░░░░░░░░░░░░░░░░░░  0.5GB                        │
│  Semantic Cache    ██░░░░░░░░░░░░░░░░░░░  0.5GB                        │
│  Context Buffer    ████░░░░░░░░░░░░░░░░░  1.0GB                        │
│  System            ███░░░░░░░░░░░░░░░░░░  0.5GB                        │
│  Buffer (safety)   ████████████████░░░░░  7.5GB                        │
│  ───────────────────────────────────────────────                       │
│  TOPLAM            ████████████████████░  ~16GB                        │
│                                                                         │
│  Özellik Durumu:                                                       │
│  ─────────────────                                                     │
│  ✅ Interactive Chat                                                   │
│  ✅ Multi-turn Conversation                                            │
│  ✅ Context Memory                                                     │
│  ✅ Full RAG System                                                    │
│  ✅ File Operations                                                    │
│  ✅ Semantic Cache                                                     │
│  ✅ Parallel Search (2 threads)                                        │
│  ✅ Large File Analysis                                                │
│  ✅ Session Persistence                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🔵 POWER MODE (24GB - 48GB Available)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    POWER MODE - 32GB FREE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  "Profesyonel kullanım, büyük projeler"                                │
│                                                                         │
│  Ayarlar:                                                               │
│  ────────                                                               │
│  Context Window    : 65,536 tokens (64K)                               │
│  Quantization      : 8-bit (higher quality)                            │
│  Batch Size        : 8                                                 │
│  RAG Cache         : 2GB                                               │
│  KV Cache          : 8GB                                               │
│  Embedding         : CodeBERT-large (1GB)                              │
│  Concurrent Ops    : 4                                                 │
│  Model Loading     : Eager (full preload)                              │
│  Compaction        : Relaxed (son 10 mesaj tam)                        │
│                                                                         │
│  Bellek Dağılımı:                                                      │
│  ─────────────────                                                     │
│  Model (8-bit)     ████████░░░░░░░░░░░░░  2.5GB                        │
│  CodeBERT-large    ████░░░░░░░░░░░░░░░░░  1.0GB                        │
│  KV Cache          ██████████████████░░░  8.0GB                        │
│  RAG Cache         ████████░░░░░░░░░░░░░  2.0GB                        │
│  Semantic Cache    ████░░░░░░░░░░░░░░░░░  1.0GB                        │
│  Context Buffer    ████░░░░░░░░░░░░░░░░░  1.0GB                        │
│  Multi-file Buffer ████░░░░░░░░░░░░░░░░░  1.0GB                        │
│  System            ██░░░░░░░░░░░░░░░░░░░  0.5GB                        │
│  Buffer (safety)   ███████████████████░░  15GB                         │
│  ───────────────────────────────────────────────                       │
│  TOPLAM            ████████████████████░  ~32GB                        │
│                                                                         │
│  Özellik Durumu:                                                       │
│  ─────────────────                                                     │
│  ✅ Tüm Comfort Mode özellikleri                                       │
│  ✅ 8-bit quantization (better quality)                                │
│  ✅ Extended context (64K tokens)                                      │
│  ✅ Multi-file analysis (4 files parallel)                             │
│  ✅ Large project support (monorepos)                                  │
│  ✅ Advanced semantic cache                                            │
│  ✅ Parallel inference                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🟣 BEAST MODE (48GB+ Available)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BEAST MODE - 64GB+ FREE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  "Sınırsız güç, enterprise projeler"                                   │
│                                                                         │
│  Ayarlar:                                                               │
│  ────────                                                               │
│  Context Window    : 131,072 tokens (128K)                             │
│  Quantization      : 8-bit (max quality)                               │
│  Batch Size        : 16                                                │
│  RAG Cache         : 8GB (entire index in RAM)                         │
│  KV Cache          : 16GB                                              │
│  Embedding         : CodeBERT-large (1GB)                              │
│  Concurrent Ops    : 8                                                 │
│  Model Loading     : Full eager + warm cache                           │
│  Compaction        : Minimal (son 20 mesaj tam)                        │
│  Secondary Model   : 7B model available for complex tasks              │
│                                                                         │
│  Ek Özellikler:                                                        │
│  ───────────────                                                       │
│  ✅ 128K context (tüm proje tek seferde)                               │
│  ✅ Multiple models (1.3B + 7B switch)                                 │
│  ✅ Full project indexing in RAM                                       │
│  ✅ 8 parallel operations                                              │
│  ✅ Zero-latency RAG (all cached)                                      │
│  ✅ Background indexing                                                │
│  ✅ Speculative execution                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Profil Karşılaştırma Tablosu

| Özellik | 🟡 8GB | 🟢 16GB | 🔵 32GB | 🟣 64GB+ |
|---------|--------|---------|---------|----------|
| **Context** | 16K | 32K | 64K | 128K |
| **Quantization** | 4-bit | 4-bit | 8-bit | 8-bit |
| **Batch Size** | 1 | 4 | 8 | 16 |
| **RAG Cache** | 200MB | 500MB | 2GB | 8GB |
| **KV Cache** | 2GB | 4GB | 8GB | 16GB |
| **Embedding** | Small | Base | Large | Large |
| **Concurrent** | 1 | 2 | 4 | 8 |
| **Compaction** | Aggressive | Smart | Relaxed | Minimal |
| **Response Quality** | Good | Great | Excellent | Maximum |
| **Speed** | ~3 tok/s | ~5 tok/s | ~8 tok/s | ~12 tok/s |

## Dinamik RAM Yönetimi (Smart Allocation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DİNAMİK RAM YÖNETİMİ                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  QuirkLLM sadece toplam RAM'e değil, anlık BOŞTA olan RAM'e bakar.     │
│  Diğer uygulamalarınızın (Chrome, Docker vb.) alanına girmez.          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Total RAM: 32GB  |  Used: 20GB  |  Available: 12GB             │   │
│  │  Decision: COMFORT MODE (fits in 12GB)                          │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  < 50%  : Normal operation                                      │   │
│  │  50-70% : Optimize mode (reduce cache)                          │   │
│  │  70-85% : Defensive mode (aggressive compaction)                │   │
│  │  > 85%  : Emergency mode (offload to disk)                      │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Örnek Senaryo:                                                        │
│  ──────────────                                                        │
│  1. Kullanıcı büyük dosya açar                                        │
│  2. RAM 75%'e çıkar                                                   │
│  3. QuirkLLM otomatik:                                                │
│     • Eski RAG cache'i temizler                                       │
│     • Compaction'ı artırır                                            │
│     • KV cache'i optimize eder                                        │
│  4. RAM 55%'e düşer                                                   │
│  5. Kullanıcı hiçbir şey fark etmez ✨                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Manuel Profil Override

```bash
# Otomatik (önerilen)
$ quirkllm

# Manuel override
$ quirkllm --profile survival    # 8GB modunda çalıştır
$ quirkllm --profile comfort     # 16GB modunda çalıştır
$ quirkllm --profile power       # 32GB modunda çalıştır
$ quirkllm --profile beast       # 64GB modunda çalıştır

# Veya session içinde
> /profile power
⚡ Switched to POWER MODE
   Context: 32K → 64K
   Quantization: 4-bit → 8-bit
   This may use more RAM.
```

---

# 🎮 4 ÇALIŞMA MODU (Shift+Tab)

QuirkLLM, farklı çalışma stilleri için 4 farklı mod sunar. Modlar arasında `Shift+Tab` ile anında geçiş yapabilirsiniz.

## 1. 💬 CHAT MODE (Varsayılan)
Klasik, güvenli, onaylı mod. Her kritik işlem öncesi sizden onay ister.
*   **Güvenlik:** Yüksek
*   **Hız:** Normal
*   **Kullanım:** Günlük kodlama, öğrenme, debug.

## 2. 🚀 YAMI MODE (Yamac Mode / YOLO)
**"Auto-Accept"** modu. Onay sormaz, kodu yazar, testi çalıştırır, dosyayı düzeltir.
*   **Güvenlik:** Düşük (Dikkatli olun!)
*   **Hız:** Maksimum
*   **Kullanım:** Hızlı prototipleme, güvenilen scriptler, "biliyorum ne yaptığımı" anları.
*   **Özellik:** `rm -rf` gibi çok riskli komutlar hariç her şeye "YES" der.

```bash
[YAMI] > fix all linter errors
🤖 Fixing 12 files...
✓ Done.
🤖 Running tests...
✓ Passed.
(Sıfır kullanıcı müdahalesi)
```

## 3. 📐 PLAN MODE (Architect)
Sadece planlama yapar. Kod yazmaz, dosya değiştirmez.
*   **Güvenlik:** Maksimum (Read-only)
*   **Çıktı:** `TODO.md`, `ARCHITECTURE.md`, Mermaid diyagramları.
*   **Kullanım:** Büyük refactoring öncesi, proje analizi, dokümantasyon.

```bash
[PLAN] > refactor auth system
🤖 Analyzing current auth flow...
📋 Created plan: .quirkllm/plans/auth-refactor.md
1. Create TokenService
2. Update UserSchema
3. Migrate existing users
...
(Hiçbir kod değişmedi, sadece plan oluşturuldu)
```

## 4. 👻 GHOST MODE (Watcher)
Siz kodunuzu IDE'nizde yazarken arka planda sessizce çalışır. Dosyayı kaydettiğiniz an (`Ctrl+S`) devreye girer ve "Arkanızı kollar".
*   **Aktiflik:** Pasif / Arka Plan
*   **İşlev:** Değişiklik analizi, hata yakalama, etki analizi (Impact Analysis).
*   **Bildirim:** Terminal uyarısı.

```bash
[GHOST] > Watching for changes...
(Kullanıcı User.ts dosyasını kaydeder)
👻 Pssst! `User.ts` değişikliği `AuthService.ts` dosyasını kırdı.
   > Fix it? (y/n)
```

---

# 🆚 NEDEN QUIRKLLM?

## Karşılaştırma

| Özellik | Claude Code CLI | Gemini CLI | GitHub Copilot | **QuirkLLM** |
|---------|-----------------|------------|----------------|--------------|
| **Fiyat** | $20/ay + API | Ücretsiz (limitli) | $10-19/ay | **Ücretsiz** |
| **Lokal** | ❌ | ❌ | ❌ | **✅** |
| **Offline** | ❌ | ❌ | ❌ | **✅** |
| **GPU Gerekli** | - | - | - | **Opsiyonel (Varsa Hızlanır)** |
| **Gizlilik** | Bulut | Bulut | Bulut | **%100 Lokal** |
| **Açık Kaynak** | ❌ | ❌ | ❌ | **✅** |
| **Interactive Chat** | ✅ | ✅ | ❌ | **✅** |
| **Context** | 200K | 1M | Kısıtlı | **16K-128K*** |
| **Min RAM** | - | - | - | **8GB** |
| **Adaptive** | ❌ | ❌ | ❌ | **✅** |

*RAM'e göre dinamik

## Hedef Kitle

- 💻 GPU'su olmayan geliştiriciler
- 🌍 İnternet erişimi kısıtlı olanlar
- 🔒 Kodunu buluta göndermek istemeyenler
- 💰 Aylık abonelik ödeyemeyenler/ödemek istemeyenler
- 🎓 Öğrenciler ve hobi geliştiriciler

---

# ✨ ÖZELLİKLER

## 1. Interactive Chat Mode

```
$ quirkllm

🔍 System: 16GB RAM detected
⚡ Profile: COMFORT MODE (32K context)

╭─────────────────────────────────────────────────────────────────╮
│  🚀 QuirkLLM v1.0.0 - Interactive Coding Assistant              │
│  📁 Project: ~/my-react-app (React 18, TypeScript, yarn)        │
│  🧠 Context: 32K tokens | RAM: 3.2GB used                       │
╰─────────────────────────────────────────────────────────────────╯

> merhaba
🤖 Merhaba! Ben QuirkLLM. React 18 + TypeScript projenizde 
   nasıl yardımcı olabilirim?

> bu projede neler var?
🤖 Projenizi analiz ediyorum...

📁 my-react-app
├── Framework: React 18.2.0 + TypeScript 5.0
├── Bundler: Vite 5.0
├── Styling: TailwindCSS 3.4
├── State: Zustand 4.5
├── Package Manager: yarn (yarn.lock detected)
├── 23 components
├── 12 hooks
└── 8 API endpoints

Ne üzerinde çalışmak istersin?
```

## 2. Context-Aware Conversations

```
> create a button component
✓ Created Button.tsx

> make it larger           # "it" = Button.tsx
✓ Updated Button.tsx

> add red variant          # hala Button.tsx
✓ Updated Button.tsx

> now do the same for Card  # "same" = aynı değişiklikler
✓ Updated Card.tsx
```

## 3. Smart Package Manager Detection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PAKET YÖNETİCİSİ TESPİTİ                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  QuirkLLM projenizde hangi paket yöneticisi kullandığınızı             │
│  otomatik algılar ve ona göre komut çalıştırır:                        │
│                                                                         │
│  Algılama Sırası:                                                      │
│  ─────────────────                                                     │
│  1. bun.lockb        → bun add <package>                               │
│  2. pnpm-lock.yaml   → pnpm add <package>                              │
│  3. yarn.lock        → yarn add <package>                              │
│  4. package-lock.json → npm install <package>                          │
│  5. Hiçbiri yoksa    → Kullanıcıya sor                                 │
│                                                                         │
│  Örnek:                                                                │
│  ───────                                                               │
│  > add framer motion for animations                                    │
│                                                                         │
│  🔍 Package manager: yarn (yarn.lock detected)                         │
│                                                                         │
│  ✓ Running: yarn add framer-motion                                     │
│  ✓ Updated: package.json                                               │
│  ✓ Updated: src/components/Button.tsx                                  │
│                                                                         │
│  Python projeleri için:                                                │
│  ──────────────────────                                                │
│  1. poetry.lock      → poetry add <package>                            │
│  2. Pipfile.lock     → pipenv install <package>                        │
│  3. requirements.txt → pip install <package>                           │
│  4. pyproject.toml   → pip install <package>                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. Multi-Turn Reasoning

```
> fix all typescript errors in this project

🤖 Scanning project for TypeScript errors...

Found 7 errors in 4 files:

1. src/utils/api.ts:23 - Type 'string' not assignable to 'number'
2. src/utils/api.ts:45 - Property 'data' does not exist
3. src/components/Card.tsx:12 - Missing required prop 'title'
...

Should I fix all of them? (yes/no/select)

> yes

✓ Fixed src/utils/api.ts (2 errors)
✓ Fixed src/components/Card.tsx (1 error)
✓ Fixed src/hooks/useAuth.ts (3 errors)
✓ Fixed src/pages/Home.tsx (1 error)

All 7 errors fixed! Run `yarn typecheck` to verify.
```

## 5. Smart Compact Mode

```
> /compact on
📦 Compact mode enabled. Responses will be shorter.

> create a modal component
✓ Modal.tsx (32 lines)

> /compact off
📖 Verbose mode enabled.
```

## 6. Session Persistence

```
# Pazartesi
$ quirkllm
> working on auth system
...
> /quit
💾 Session saved.

# Salı
$ quirkllm
🔄 Restored previous session (auth system)
   Last message: "implementing JWT refresh"
   Context: 12,450/32,768 tokens

> continue where we left off
🤖 Last time we were implementing JWT refresh tokens. 
   You had created authService.ts. Should I continue?
```

## 7. Adaptive Performance

```
> /status

╭─ System Status ─────────────────────────────────────────────────────────╮
│                                                                         │
│  Profile     : COMFORT MODE (16GB system)                              │
│  RAM Usage   : 4.2GB / 16GB (26%) ████░░░░░░░░░░░░                     │
│  Context     : 8,421 / 32,768 tokens (25%) █████░░░░░░░░░░░            │
│                                                                         │
│  Performance:                                                          │
│  ─────────────                                                         │
│  Inference   : 5.2 tokens/sec                                          │
│  Cache Hit   : 73%                                                     │
│  RAG Latency : 45ms                                                    │
│                                                                         │
│  Session:                                                              │
│  ─────────                                                             │
│  Messages    : 12 (5 compacted)                                        │
│  Files       : 3 active (Button.tsx, Card.tsx, api.ts)                 │
│  Duration    : 23 minutes                                              │
│                                                                         │
╰─────────────────────────────────────────────────────────────────────────╯

## 8. 📚 Knowledge Eater (Lokal Doküman Yiyici)

İnternet yok mu? Özel dokümantasyon mu var? Sorun değil. QuirkLLM istediğiniz kaynağı "yutup" lokal vektör veritabanına ekler.

```bash
> learn --url https://docs.astro.build
🤖 Crawling docs.astro.build...
✓ Downloaded 142 pages
✓ Generated embeddings (CodeBERT)
✓ Added to local vector store
🧠 Now I know everything about Astro!

> learn --pdf ./sirket-kurallari.pdf
✓ Parsed PDF
✓ Indexed 45 pages
```

## 9. 🔌 Neural Link (MCP Server Support)

QuirkLLM sadece bir asistan değil, aynı zamanda bir **Model Context Protocol (MCP)** sunucusudur. Claude Desktop, Cursor veya diğer MCP uyumlu araçlar, QuirkLLM'in lokal indeksine ve yeteneklerine bağlanabilir.

```bash
> /mcp start
🔌 MCP Server running on ws://localhost:3000
   • Capabilities: Resources, Tools, Prompts
   • Connected: Claude Desktop can now see your local repo!
```
```

---

# 💡 KULLANIM ÖRNEKLERİ

## Senaryo 1: Yeni Proje Başlatma

```
$ cd my-new-project
$ quirkllm

🔍 System: 16GB RAM → COMFORT MODE
📁 Empty directory detected

> setup a react typescript project with tailwind and zustand

🤖 I'll set up a modern React project for you.

Which package manager do you prefer?
  1. npm
  2. yarn (recommended)
  3. pnpm
  4. bun

> 2

✓ Initialized with Vite + React + TypeScript
✓ Added TailwindCSS configuration
✓ Added Zustand store boilerplate
✓ Created folder structure

📁 Project structure:
├── src/
│   ├── components/
│   ├── hooks/
│   ├── store/
│   │   └── index.ts (Zustand store)
│   ├── App.tsx
│   └── main.tsx
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts

Run `yarn install` then `yarn dev` to start!
```

## Senaryo 2: Bug Fixing Session

```
> the app crashes when clicking submit button

🤖 I'll investigate. Let me check the submit handlers...

🔍 Found 3 files with submit handlers:
   • ContactForm.tsx (line 23)
   • LoginForm.tsx (line 45)
   • SignupForm.tsx (line 67)

Which one is causing the crash?

> ContactForm

🔍 Analyzing src/components/ContactForm.tsx...

Found potential issues:

1. Line 23: `onSubmit` is async but errors aren't caught
2. Line 31: `setLoading(false)` runs before API completes
3. Line 45: No validation before submit

Should I fix all of these?

> yes

✓ Added try-catch wrapper (line 23)
✓ Fixed loading state race condition (line 31)
✓ Added form validation (line 45)

The form should work now. Test it?
```

## Senaryo 3: Code Review & Refactoring

```
> review src/utils/helpers.ts

🔍 Analyzing helpers.ts (156 lines)...

📋 Code Review Report:

│ Severity │ Count │ Issues                              │
├──────────┼───────┼─────────────────────────────────────┤
│ 🔴 Critical │  1  │ SQL injection vulnerability (L45)   │
│ 🟡 Warning  │  3  │ Long function, magic number, unused │
│ 🔵 Info     │  2  │ Optional chaining, Map suggestion   │

Details:
─────────
🔴 Line 45: buildQuery() is vulnerable to SQL injection
🟡 Line 12: processData() is 87 lines (recommend: <30)
🟡 Line 34: Magic number 86400 (use named constant)
🟡 Line 67: Variable 'temp' is declared but never used

Should I fix any of these?

> fix critical and warnings

✓ Fixed SQL injection (parameterized query)
✓ Split processData into 3 functions
✓ Added SECONDS_PER_DAY constant
✓ Removed unused variable

Run `/diff` to see all changes.
```

## Senaryo 4: Learning & Explanation

```
> explain how useCallback works, I keep getting it wrong

🤖 Let me explain with examples from your codebase.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

useCallback = memoize a FUNCTION reference

Without useCallback:
─────────────────────
const handleClick = () => { ... }
// ↑ New function created EVERY render
// ↑ Child components re-render unnecessarily

With useCallback:
─────────────────────
const handleClick = useCallback(() => { ... }, [deps])
// ↑ Same function reference unless deps change
// ↑ Child components skip re-render

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Found in your code (SearchBar.tsx line 12):

❌ Current (recreates every render):
   const handleSearch = (query) => {
     fetchResults(query);
   };

✅ Fixed:
   const handleSearch = useCallback((query) => {
     fetchResults(query);
   }, [fetchResults]);

Want me to fix it?

> yes

✓ Updated SearchBar.tsx
```

## Senaryo 5: Test Generation

```
> write tests for UserService

🔍 Analyzing src/services/UserService.ts...

Found 5 public methods:
• getUser(id)
• createUser(data)
• updateUser(id, data)
• deleteUser(id)
• searchUsers(query)

✓ Created src/services/__tests__/UserService.test.ts

Test Summary:
─────────────
UserService
├── getUser
│   ├── ✓ returns user when found
│   ├── ✓ returns null when not found
│   └── ✓ handles database errors
├── createUser
│   ├── ✓ creates user with valid data
│   ├── ✓ throws on duplicate email
│   └── ✓ validates required fields
└── ... (15 more tests)

Total: 23 tests generated

Run `yarn test UserService` to execute.
```

---

# 🔧 TEKNİK DETAYLAR

## Sistem Gereksinimleri

| Profil | Min. Boş RAM | Disk | CPU | Deneyim |
|--------|--------------|------|-----|---------|
| 🟡 Survival | **~8GB** | 10GB | 4 core | Çalışır |
| 🟢 Comfort | **~16GB** | 15GB | 6 core | İdeal |
| 🔵 Power | **~32GB** | 20GB | 8 core | Profesyonel |
| 🟣 Beast | **~64GB+** | 30GB | 12 core | Maksimum |

**GPU Zorunlu Değildir!** Ancak NVIDIA (CUDA) veya Apple Silicon (Metal) tespit edilirse, QuirkLLM otomatik olarak **Hybrid Inference** moduna geçer ve yükü GPU'ya yıkarak 10x-50x hız artışı sağlar.

## Model Spesifikasyonları

| Özellik | 4-bit | 8-bit |
|---------|-------|-------|
| Base Model | DeepSeek Coder 1.3B | DeepSeek Coder 1.3B |
| Disk Size | 700MB | 1.4GB |
| RAM Usage | 1.5GB | 2.5GB |
| Quality | Great | Excellent |
| Speed | Faster | Slower |
| Min RAM | 8GB | 16GB |

## Adaptive Context Length

```python
# QuirkLLM otomatik context hesaplama

def calculate_context_length(available_ram_gb, quantization):
    """RAM'e göre optimal context length hesapla"""
    
    # Sabit kullanımlar
    model_ram = 1.5 if quantization == "4bit" else 2.5
    embedding_ram = 0.5
    base_overhead = 1.0
    
    # KV Cache için kalan RAM
    available_for_kv = available_ram_gb - model_ram - embedding_ram - base_overhead
    
    # Her 1K context ≈ 250MB KV cache (4-bit için)
    # Her 1K context ≈ 400MB KV cache (8-bit için)
    mb_per_1k = 250 if quantization == "4bit" else 400
    
    max_context = int((available_for_kv * 1024) / mb_per_1k) * 1024
    
    # Profil limitleri
    limits = {
        8:  16384,   # 16K max for 8GB
        16: 32768,   # 32K max for 16GB
        32: 65536,   # 64K max for 32GB
        64: 131072,  # 128K max for 64GB+
    }
    
    return min(max_context, limits.get(available_ram_gb, 131072))
```

---

# 🏗️ MİMARİ

## Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QuirkLLM MİMARİSİ                               │
└─────────────────────────────────────────────────────────────────────────┘

                              KULLANICI
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SYSTEM DETECTOR (Başlangıç)                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • RAM Detection (psutil)                                       │   │
│  │  • CPU Core Count                                               │   │
│  │  • Available Disk Space                                         │   │
│  │  • Profile Selection (survival/comfort/power/beast)             │   │
│  │  • Dynamic Configuration Loading                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INTERACTIVE CLI                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • REPL Loop (Read-Eval-Print)                                  │   │
│  │  • Command Parser (/help, /compact, /status, /profile...)       │   │
│  │  • Session Manager (save/load/restore)                          │   │
│  │  • Rich Terminal UI (colors, boxes, syntax highlighting)        │   │
│  │  • RAM Monitor (background thread)                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PROJECT ANALYZER                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Framework Detection (React, Vue, Next, Expo...)              │   │
│  │  • Package Manager Detection (npm/yarn/pnpm/bun/poetry/pip)     │   │
│  │  • Dependency Analysis                                          │   │
│  │  • File Structure Mapping                                       │   │
│  │  • Active File Tracking                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CONVERSATION ENGINE                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Context Window Manager (16K-128K adaptive)                   │   │
│  │  • Conversation History (with smart compaction)                 │   │
│  │  • Multi-Turn Reasoning                                         │   │
│  │  • Reference Resolution ("it", "that", "the component")         │   │
│  │  • Intent Detection (create/fix/explain/refactor...)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────┐
│    ADAPTIVE INFERENCE ENGINE  │ │        ADAPTIVE RAG SYSTEM            │
│  ┌─────────────────────────┐  │ │  ┌─────────────────────────────────┐ │
│  │  Survival Mode (8GB)    │  │ │  │  Survival: 200MB cache, basic   │ │
│  │  • 4-bit aggressive     │  │ │  │  Comfort:  500MB cache, full    │ │
│  │  • Batch size: 1        │  │ │  │  Power:    2GB cache, parallel  │ │
│  │  • Sequential only      │  │ │  │  Beast:    8GB cache, instant   │ │
│  ├─────────────────────────┤  │ │  └─────────────────────────────────┘ │
│  │  Comfort Mode (16GB)    │  │ │                                       │
│  │  • 4-bit balanced       │  │ │  ┌─────────────────────────────────┐ │
│  │  • Batch size: 4        │  │ │  │  Components (18):               │ │
│  │  • 2 concurrent ops     │  │ │  │  • LanceDB Vector Store         │ │
│  ├─────────────────────────┤  │ │  │  • CodeBERT Embeddings          │ │
│  │  Power Mode (32GB)      │  │ │  │  • Hybrid Search                │ │
│  │  • 8-bit quality        │  │ │  │  • Semantic Cache               │ │
│  │  • Batch size: 8        │  │ │  │  • Context Compression          │ │
│  │  • 4 concurrent ops     │  │ │  │  • ...13 more                   │ │
│  ├─────────────────────────┤  │ │  └─────────────────────────────────┘ │
│  │  Beast Mode (64GB+)     │  │ │                                       │
│  │  • 8-bit maximum        │  │ └───────────────────────────────────────┘
│  │  • Batch size: 16       │  │
│  │  • 8 concurrent ops     │  │
│  │  • 7B model available   │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CORE MODEL                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    DeepSeek Coder 1.3B                          │   │
│  │            (Fine-tuned + Adaptively Quantized)                  │   │
│  │                                                                  │   │
│  │  Fine-tuning (30 components):                                   │   │
│  │  • Multi-turn Conversation    • Framework-Aware                 │   │
│  │  • Instruction Following      • Error Recovery                  │   │
│  │  • Code Generation            • Test Generation                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        OUTPUT HANDLER                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Response Formatting                                          │   │
│  │  • Syntax Highlighting                                          │   │
│  │  • Diff Generation                                              │   │
│  │  • Package Manager Commands (yarn/npm/pnpm/bun)                 │   │
│  │  • File Creation/Modification                                   │   │
│  │  • Confirmation Prompts                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## CLI-First Felsefesi

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CLI-FIRST FELSEFESİ                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CLI = TÜM AKIL (78 bileşen)       IDE/GUI = SADECE WRAPPER           │
│   ─────────────────────────────     ──────────────────────────         │
│   • RAM detection & adaptation      • CLI'ı subprocess olarak çağırır │
│   • Tüm conversation logic          • Sadece UI katmanı               │
│   • Tüm inference optimizasyonları  • Sıfır ek mantık                  │
│   • Tüm RAG sistemi                 • Sıfır ek bileşen                 │
│   • %100 fonksiyonellik             • %100 görsellik                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 🔧 78 BİLEŞEN

## Bileşen Dağılımı

| Katman | Sayı | RAM Adaptif? |
|--------|------|--------------|
| A. Inference Optimizasyonu | 15 | ✅ |
| B. Bellek/CPU Optimizasyonu | 15 | ✅ |
| C. RAG Sistemi | 18 | ✅ |
| D. Fine-tuning Stratejisi | 30 | ❌ (eğitim zamanı) |
| **TOPLAM** | **78** | |

## A. Inference Optimizasyonu (15) - RAM Adaptif

| # | Bileşen | 8GB | 16GB | 32GB | 64GB+ |
|---|---------|-----|------|------|-------|
| A1 | Early Exit | ✅ Aggressive | ✅ Balanced | ✅ Relaxed | ⚪ Off |
| A2 | Layer Cache | ✅ 2 layers | ✅ 4 layers | ✅ 8 layers | ✅ All |
| A3 | Speculative Decoding | ⚪ Off | ✅ On | ✅ On | ✅ Aggressive |
| A4 | Dynamic Quantization | ✅ 4-bit | ✅ 4-bit | ✅ 8-bit | ✅ 8-bit |
| A5 | Token Pruning | ✅ High | ✅ Medium | ✅ Low | ⚪ Off |
| A6 | KV-Cache Optimization | ✅ Aggressive | ✅ Normal | ✅ Relaxed | ✅ Full |
| A7 | Batch Inference | ✅ 1 | ✅ 4 | ✅ 8 | ✅ 16 |
| A8 | Continuous Batching | ⚪ Off | ✅ On | ✅ On | ✅ On |
| A9 | Tensor Parallelism | ⚪ Off | ⚪ Off | ✅ 2-way | ✅ 4-way |
| A10 | Activation Checkpoint | ✅ All | ✅ Half | ⚪ Off | ⚪ Off |
| A11 | Flash Decoding | ✅ On | ✅ On | ✅ On | ✅ On |
| A12 | Paged Attention | ✅ On | ✅ On | ✅ On | ✅ On |
| A13 | Grouped Query Attention | ✅ On | ✅ On | ✅ On | ✅ On |
| A14 | Sliding Window Attention | ✅ 2K | ✅ 4K | ✅ 8K | ✅ 16K |
| A15 | Tree Attention | ⚪ Off | ✅ Basic | ✅ Full | ✅ Full |

## B. Bellek/CPU Optimizasyonu (15) - RAM Adaptif

| # | Bileşen | 8GB | 16GB | 32GB | 64GB+ |
|---|---------|-----|------|------|-------|
| B1 | Memory-Mapped Files | ✅ Full | ✅ Partial | ⚪ Off | ⚪ Off |
| B2 | Prefetching | ✅ Minimal | ✅ Normal | ✅ Aggressive | ✅ Full |
| B3 | Sparse Attention | ✅ High | ✅ Medium | ✅ Low | ⚪ Off |
| B4 | Gradient Checkpointing | ✅ All | ✅ Half | ⚪ Off | ⚪ Off |
| B5 | CPU Offloading | ✅ On | ⚪ Off | ⚪ Off | ⚪ Off |
| B6 | Disk Offloading | ✅ Ready | ⚪ Off | ⚪ Off | ⚪ Off |
| B7 | NUMA Awareness | ✅ On | ✅ On | ✅ On | ✅ On |
| B8 | Cache-Line Optimization | ✅ On | ✅ On | ✅ On | ✅ On |
| B9 | Memory Pool | ✅ Small | ✅ Medium | ✅ Large | ✅ XL |
| B10 | Zero-Copy Loading | ✅ On | ✅ On | ✅ On | ✅ On |
| B11 | Lazy Loading | ✅ Full | ✅ Partial | ⚪ Off | ⚪ Off |
| B12 | Weight Streaming | ✅ On | ⚪ Off | ⚪ Off | ⚪ Off |
| B13 | Async I/O | ✅ On | ✅ On | ✅ On | ✅ On |
| B14 | mmap + madvise | ✅ On | ✅ On | ✅ On | ✅ On |
| B15 | Huge Pages | ⚪ Off | ✅ On | ✅ On | ✅ On |

## C. RAG Sistemi (18) - RAM Adaptif

| # | Bileşen | 8GB | 16GB | 32GB | 64GB+ |
|---|---------|-----|------|------|-------|
| C1 | LanceDB | ✅ Disk | ✅ Hybrid | ✅ RAM | ✅ Full RAM |
| C2 | Semantic Search | ✅ Basic | ✅ Full | ✅ Full | ✅ Full |
| C3 | Hybrid Search | ⚪ Off | ✅ On | ✅ On | ✅ On |
| C4 | Reranking | ⚪ Off | ✅ Top-5 | ✅ Top-10 | ✅ Top-20 |
| C5 | Context Compression | ✅ High | ✅ Medium | ✅ Low | ⚪ Off |
| C6 | Semantic Cache | ✅ 50MB | ✅ 200MB | ✅ 1GB | ✅ 4GB |
| C7 | Query Expansion | ⚪ Off | ✅ On | ✅ On | ✅ On |
| C8 | Query Decomposition | ⚪ Off | ✅ Basic | ✅ Full | ✅ Full |
| C9 | HyDE | ⚪ Off | ⚪ Off | ✅ On | ✅ On |
| C10 | Multi-hop Retrieval | ⚪ Off | ✅ 2-hop | ✅ 3-hop | ✅ 5-hop |
| C11 | Parent-Child Chunking | ⚪ Off | ✅ On | ✅ On | ✅ On |
| C12 | Sliding Window Chunk | ✅ On | ✅ On | ✅ On | ✅ On |
| C13 | Code-Aware Chunking | ✅ Basic | ✅ Full | ✅ Full | ✅ Full |
| C14 | Metadata Filtering | ✅ On | ✅ On | ✅ On | ✅ On |
| C15 | Version Filtering | ✅ On | ✅ On | ✅ On | ✅ On |
| C16 | Freshness Scoring | ✅ On | ✅ On | ✅ On | ✅ On |
| C17 | Relevance Feedback | ⚪ Off | ✅ On | ✅ On | ✅ On |
| C18 | Auto-Indexing | ⚪ Off | ✅ Idle | ✅ Background | ✅ Realtime |

## D. Fine-tuning Stratejisi (30)

| # | Bileşen | # | Bileşen |
|---|---------|---|---------|
| D1 | Instruction Following | D16 | Security Patterns |
| D2 | Multi-turn Conversation | D17 | Accessibility |
| D3 | FIM Training | D18 | i18n Patterns |
| D4 | Error Recovery | D19 | State Management |
| D5 | Multi-file Context | D20 | API Integration |
| D6 | Test Generation | D21 | Database Patterns |
| D7 | Docstring→Code | D22 | Type Inference |
| D8 | Code→Docstring | D23 | Error Handling |
| D9 | Refactoring | D24 | Async Patterns |
| D10 | Framework-Aware | D25 | Component Patterns |
| D11 | Version-Aware | D26 | Hook Patterns |
| D12 | Best Practices | D27 | Testing Patterns |
| D13 | Debug Reasoning | D28 | CI/CD Awareness |
| D14 | Code Review | D29 | Package.json Aware |
| D15 | Performance Opt | D30 | Monorepo Aware |

---

# 💻 CLI KOMUTLARI

## Slash Komutları

| Komut | Açıklama | Örnek |
|-------|----------|-------|
| `/help` | Yardım menüsü | `/help` |
| `/status` | Sistem durumu (RAM, context, cache) | `/status` |
| `/profile` | Profil değiştir/göster | `/profile power` |
| `/compact` | Compact mode toggle | `/compact on` |
| `/verbose` | Verbose mode toggle | `/verbose on` |
| `/context` | Mevcut context'i göster | `/context` |
| `/clear` | Konuşmayı temizle | `/clear` |
| `/reset` | Tüm state'i sıfırla | `/reset` |
| `/save` | Session'ı kaydet | `/save mysession` |
| `/load` | Session yükle | `/load mysession` |
| `/sessions` | Kayıtlı session'ları listele | `/sessions` |
| `/mode` | Quantization mode | `/mode 8bit` |
| `/offline` | Offline mode toggle | `/offline` |
| `/diff` | Son değişiklikleri göster | `/diff` |
| `/undo` | Son değişikliği geri al | `/undo` |
| `/files` | Değiştirilen dosyaları listele | `/files` |
| `/tree` | Proje yapısını göster | `/tree` |
| `/config` | Ayarları göster/değiştir | `/config` |
| `/quit` | Çıkış | `/quit` |

## Özel Prefixler

| Prefix | Açıklama | Örnek |
|--------|----------|-------|
| `@file` | Dosya referansı | `@src/App.tsx explain this` |
| `#line` | Satır referansı | `fix error on #23` |
| `!command` | Shell komutu | `!yarn test` |

---

# 🧠 CONTEXT YÖNETİMİ

## Adaptive Context Window

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE CONTEXT WINDOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  8GB RAM (16K tokens):                                                 │
│  ════════════════════                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ System Prompt          : ~300 tokens                            │   │
│  │ Project Context        : ~500 tokens                            │   │
│  │ Active Files           : ~3000 tokens (1-2 files)               │   │
│  │ RAG Retrieved          : ~2000 tokens                           │   │
│  │ Conversation (compact) : ~6000 tokens (~20 turns)               │   │
│  │ Current Query/Response : ~4200 tokens                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  16GB RAM (32K tokens):                                                │
│  ═════════════════════                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ System Prompt          : ~500 tokens                            │   │
│  │ Project Context        : ~1000 tokens                           │   │
│  │ Active Files           : ~8000 tokens (3-4 files)               │   │
│  │ RAG Retrieved          : ~4000 tokens                           │   │
│  │ Conversation (smart)   : ~12000 tokens (~50 turns)              │   │
│  │ Current Query/Response : ~6500 tokens                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  32GB RAM (64K tokens):                                                │
│  ═════════════════════                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ System Prompt          : ~500 tokens                            │   │
│  │ Project Context        : ~2000 tokens                           │   │
│  │ Active Files           : ~20000 tokens (8-10 files)             │   │
│  │ RAG Retrieved          : ~8000 tokens                           │   │
│  │ Conversation (relaxed) : ~25000 tokens (~100 turns)             │   │
│  │ Current Query/Response : ~8500 tokens                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  64GB+ RAM (128K tokens):                                              │
│  ══════════════════════                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ System Prompt          : ~500 tokens                            │   │
│  │ Project Context        : ~5000 tokens (full analysis)           │   │
│  │ Active Files           : ~50000 tokens (entire codebase)        │   │
│  │ RAG Retrieved          : ~15000 tokens                          │   │
│  │ Conversation (minimal) : ~45000 tokens (~200 turns)             │   │
│  │ Current Query/Response : ~12500 tokens                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Smart Compaction (RAM'e Göre)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SMART COMPACTION                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  8GB (Aggressive):                                                     │
│  ─────────────────                                                     │
│  • Son 3 mesaj tam tutulur                                            │
│  • Eski mesajlar → tek satır özet                                     │
│  • Kod blokları → "Created X.tsx (45 lines)"                          │
│                                                                         │
│  16GB (Smart):                                                         │
│  ──────────────                                                        │
│  • Son 5 mesaj tam tutulur                                            │
│  • Eski mesajlar → 2-3 satır özet                                     │
│  • Önemli kararlar korunur                                            │
│                                                                         │
│  32GB (Relaxed):                                                       │
│  ───────────────                                                       │
│  • Son 10 mesaj tam tutulur                                           │
│  • Eski mesajlar → paragraf özet                                      │
│  • Kod blokları kısmen korunur                                        │
│                                                                         │
│  64GB+ (Minimal):                                                      │
│  ────────────────                                                      │
│  • Son 20 mesaj tam tutulur                                           │
│  • Çok eski mesajlar bile detaylı                                     │
│  • Neredeyse hiç bilgi kaybı yok                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 🎓 FINE-TUNING (AXOLOTL)

## Konfigürasyon

```yaml
# quirkllm_finetune.yaml

base_model: deepseek-ai/deepseek-coder-1.3b-base
model_type: AutoModelForCausalLM

load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: ./data/multi_turn_conversations.jsonl
    type: sharegpt
    conversation: conversations
  - path: ./data/code_completion.jsonl
    type: completion

chat_template: chatml
sequence_len: 4096
sample_packing: true

gradient_accumulation_steps: 8
micro_batch_size: 2
num_epochs: 3
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.1

flash_attention: true
bf16: true

output_dir: ./outputs/quirkllm-v1
wandb_project: quirkllm
```

## Training Data

| Kaynak | Miktar | Format |
|--------|--------|--------|
| GitHub Repos | 10000 | Multi-turn conversations |
| Synthetic | 50000 | GPT-4 generated |
| Stack Overflow | 20000 | Q&A pairs |
| Documentation | 5000 | Chunks |

---

# 📊 TEST SUITE

## Benchmark Hedefleri

| Benchmark | Base | Target | Neden? |
|-----------|------|--------|--------|
| HumanEval | 50% | 70% | Genel kod kalitesi |
| MBPP | 45% | 65% | Python yetkinliği |
| Multi-turn Accuracy | - | 85%+ | Ana kullanım senaryosu |
| Context Resolution | - | 90%+ | "it", "that" anlama |
| Domain (React/TS) | - | 80%+ | Hedef teknolojiler |

## Test Çalıştırma

```bash
$ quirkllm --test

╭─ QuirkLLM Test Suite ───────────────────────────────────────────────────╮
│                                                                         │
│  System: 16GB RAM (COMFORT MODE)                                       │
│                                                                         │
│  Instruction Following    ████████████████████ 95/100 (95%)            │
│  Multi-Turn Conversation  █████████████████░░░ 88/100 (88%)            │
│  Context Resolution       ████████████████████ 92/100 (92%)            │
│  Code Quality             ██████████████████░░ 91/100 (91%)            │
│  RAM Adaptation           ████████████████████ 100/100 (100%)          │
│  Package Manager          ████████████████████ 100/100 (100%)          │
│                                                                         │
│  Overall: 566/600 (94.3%) ✓ PASSED                                     │
│                                                                         │
╰─────────────────────────────────────────────────────────────────────────╯
```

---

# 📅 ROADMAP

## Özet Timeline

| Faz | Süre | İçerik |
|-----|------|--------|
| CLI (78 bileşen + RAM Adaptive) | 30 hafta | Tüm sistem |
| VS Code Extension | 4 hafta | CLI wrapper |
| GUI Application | 5 hafta | CLI wrapper |
| **TOPLAM** | **39 hafta** | **~9-10 ay** |

## CLI Fazı Detay

```
Alt-Faz A: System Detection (2 hafta)
─────────────────────────────────────
• RAM detection (psutil)
• Profile selection logic
• Dynamic configuration
• Resource monitoring

Alt-Faz B: Core CLI (3 hafta)
─────────────────────────────
• Interactive REPL
• Command parser
• Session management
• Rich terminal UI

Alt-Faz C: Project Analyzer (2 hafta)
─────────────────────────────────────
• Framework detection
• Package manager detection
• Dependency analysis

Alt-Faz D: Conversation Engine (4 hafta)
────────────────────────────────────────
• Adaptive context window
• Smart compaction (RAM-based)
• Multi-turn reasoning
• Reference resolution

Alt-Faz E: Model Integration (5 hafta)
──────────────────────────────────────
• Adaptive quantization
• RAM-based inference config
• 30 inference optimizations
• 30 memory optimizations

Alt-Faz F: RAG System (5 hafta)
───────────────────────────────
• LanceDB (RAM-adaptive)
• CodeBERT embeddings
• 18 RAG components
• Adaptive caching

Alt-Faz G: Fine-tuning (6 hafta)
────────────────────────────────
• Data collection
• Axolotl training
• 30 fine-tuning components
• Multi-turn optimization

Alt-Faz H: Output & Polish (3 hafta)
────────────────────────────────────
• File operations
• Diff generation
• Package manager integration

TOPLAM: 30 hafta
```

---

# 🚀 KURULUM

## pip ile

```bash
pip install quirkllm
```

## İlk Çalıştırma

```bash
$ quirkllm

🔍 First run - detecting system...
💾 RAM: 16GB detected
⚡ Profile: COMFORT MODE selected

📦 Downloading components...
   Model (4-bit): 700MB [████████████████████] 100%
   CodeBERT: 500MB [████████████████████] 100%
   RAG Index: 200MB [████████████████████] 100%

✅ Setup complete!

╭─────────────────────────────────────────────────────────────────╮
│  🚀 QuirkLLM v1.0.0 - Interactive Coding Assistant              │
│  💾 RAM: 16GB → COMFORT MODE (32K context)                      │
│  📁 Project: ~/current-directory                                │
╰─────────────────────────────────────────────────────────────────╯

> 
```

## Docker ile

```bash
docker run -it \
  -v $(pwd):/workspace \
  -e QUIRKLLM_RAM=16gb \
  quirkllm/quirkllm
```

## Konfigürasyon

```yaml
# ~/.quirkllm/config.yaml

# Otomatik veya manuel profil
profile: auto  # auto | survival | comfort | power | beast

# Override'lar (opsiyonel)
overrides:
  context_length: 32768  # Manuel context length
  quantization: 4bit     # 4bit | 8bit
  
# Arayüz
interface:
  compact_mode: false
  syntax_highlighting: true
  auto_save_session: true

# RAG
rag:
  enabled: true
  offline_fallback: true
```

---

# ✅ KARARLAR (27 Adet)

| # | Karar | Sonuç |
|---|-------|-------|
| 1 | Base Model | DeepSeek Coder 1.3B |
| 2 | Hedef Diller | React, TS, Python, Expo + 30 teknoloji |
| 3 | Min. Sistem | 8GB RAM |
| 4 | CLI Tipi | Interactive Chat (Claude Code CLI gibi) |
| 5 | Proje İsmi | QuirkLLM |
| 6 | Lisans | Apache 2.0 |
| 7 | Açık Kaynak | Tam açık |
| 8 | Dağıtım | pip + Docker |
| 9 | Platform | Windows + Mac + Linux |
| 10 | Offline | Hibrit (sessizce offline) |
| 11 | Embedding | CodeBERT (adaptive size) |
| 12 | Vector DB | LanceDB (RAM-adaptive) |
| 13 | Veri Çekme | GitHub API → Clone → AST |
| 14 | Repo Sayısı | 10000 |
| 15 | Güncelleme | Aylık |
| 16 | Veri Formatı | Multi-turn Conversation |
| 17 | Veri Miktarı | ~1B token |
| 18 | Quantization | 4-bit (8GB-16GB) / 8-bit (32GB+) |
| 19 | Arayüz | CLI (core) → IDE/GUI (wrapper) |
| 20 | IDE | VS Code |
| 21 | Benchmark | %70 HumanEval, %85+ Multi-turn |
| 22 | Test | Full suite |
| 23 | Hata Rapor | GitHub Issues |
| 24 | Fine-tuning | Axolotl (QLoRA) |
| 25 | **Context Length** | **16K-128K (RAM-adaptive)** |
| 26 | **RAM Profilleri** | **4 profil (8/16/32/64GB)** |
| 27 | **Paket Yöneticisi** | **Auto-detect (npm/yarn/pnpm/bun)** |

---

# 📄 LİSANS

```
Apache License 2.0
Copyright 2024 QuirkLLM
```

---

# 🎯 HEDEF TEKNOLOJİLER

**Diller:** JavaScript, TypeScript, Python, HTML, CSS, SQL

**Frontend:** React, React Native, Next.js, Vue.js, Nuxt.js, Svelte

**Mobile:** Expo, React Native CLI

**Backend:** Node.js, Express, Fastify, NestJS, Hono, FastAPI, Django

**Styling:** TailwindCSS, Styled Components, SASS/SCSS

**Animasyon:** Framer Motion, GSAP, Lottie, React Spring, Three.js

**Database:** PostgreSQL, MySQL, MongoDB, SQLite, Redis, Supabase, Firebase

**ORM:** Prisma, Drizzle, TypeORM, Mongoose

**State:** Zustand, Redux, Jotai, React Query, SWR

---

<div align="center">

# ⭐ QuirkLLM

### Lokal Claude Code CLI Alternatifi

```
$ quirkllm

🔍 16GB RAM detected → COMFORT MODE
🚀 Ready with 32K context window!

> let's build something amazing together
```

**Ücretsiz • Lokal • Akıllı • Adaptif**

8GB'dan 64GB+'a kadar - sisteminize göre otomatik optimize!

---

Made with ❤️ for developers without GPUs

</div>
