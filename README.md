# enterprise-kb-agent
基于LangGraph+双层分级RAG实现的企业内部知识库智能问答Agent，支持PDF/Word/TXT/Markdown/Excel多格式文档本地检索、联网行业资讯查询、文档总结导出，采用FAISS摘要粗筛+Chroma+BM25混合精细检索架构。


技术栈:
大模型层：Ollama + Qwen2:7b /all-minilm 本地私有化部署
检索层：FAISS 粗筛 + Chroma 向量库 + BM25 关键词混合检索
智能体框架：LangGraph 状态图、多节点路由、工具调用、重试校验
文档解析：LangChain DocumentLoader、文本分片
工程优化：Embedding 内存缓存、日志分级、异常捕获、网络流量削峰



项目目录分层说明:
enterprise_kb_agent/
├── agent/                # LangGraph智能体核心
│   ├── nodes/            # 图节点：检索节点、工具执行节点、校验节点
│   ├── routes/          # 分支路由判断：是否需要联网、是否重检索
│   ├── state.py         # 全局状态定义，跨节点传递消息
│   └── graph_builder.py # 组装完整LangGraph图工作流
├── document/             # 文档处理模块
│   ├── loader.py        # PDF/TXT/Word通用文档加载
│   ├── splitter.py      # 文本分片、摘要生成
│   └── vector_store.py  # FAISS/Chroma向量库初始化、检索
├── tools/                # 自定义工具集
│   └── agent_tools.py   # Tavily联网检索、文档导出工具封装
├── logs/                 # 运行日志目录
├── utils.py              # 通用工具：文档拼接、文本清洗
├── retriever.py          # 双层混合检索统一入口
├── prompts.py            # 全局提示词统一管理
├── settings.py           # 全局配置、缓存初始化、环境加载
├── log_config.py         # 日志格式化配置
├── main.py               # 控制台交互入口
├── env.example           # 环境配置模板（无真实密钥）
├── requirements.txt      # 项目依赖清单
├── .gitignore            # 版本忽略配置
└── README.md             # 项目完整说明文档


核心功能演示示例
举 2 个提问 + 输出案例，比如「试用期离职提前多久」，粘贴控制台问答效果，直观展示效果。




