# Streamlit Share 部署指南

## 📋 部署前准备工作

✅ 项目已经准备就绪，包含以下文件：
- `app.py` - 主应用文件（已移至根目录）
- `requirements.txt` - 依赖项文件
- `.streamlit/config.toml` - Streamlit配置文件
- `models/` - 模型文件目录
- `data/` - 数据文件目录

## 🚀 Streamlit Share 部署步骤

### 1. 上传到GitHub仓库
确保您的项目已上传到GitHub仓库，且包含所有必要文件。

### 2. 访问Streamlit Share
前往 [share.streamlit.io](https://share.streamlit.io)

### 3. 使用GitHub账号登录
点击"Sign in with GitHub"使用您的GitHub账号登录。

### 4. 部署新应用
1. 点击"New app"按钮
2. 选择您的GitHub仓库：`您的用户名/LungCancer-Prognosis-System`
3. 选择分支：通常是`main`或`master`
4. 设置主文件路径：`app.py`
5. 点击"Deploy!"

### 5. 等待部署完成
- 初次部署可能需要2-5分钟
- Streamlit Share会自动安装requirements.txt中的依赖项
- 部署完成后会显示应用的公共URL

### 6. 获取应用URL
部署成功后，您会得到一个类似这样的URL：
```
https://share.streamlit.io/您的用户名/lungcancer-prognosis-system/main/app.py
```

## ⚙️ 配置说明

### 文件结构
```
LungCancer-Prognosis-System/
├── app.py                    # 主应用文件
├── requirements.txt          # Python依赖项
├── .streamlit/
│   └── config.toml          # Streamlit配置
├── models/
│   ├── HROS_simple_model.pkl
│   └── HRPFS_simple_model.pkl
├── data/
│   ├── sample_OS_data.csv
│   ├── sample_PFS_data.csv
│   └── ...
└── README.md
```

### 重要配置项
- **最大文件上传大小**: 200MB
- **主题颜色**: #667eea (蓝紫色)
- **背景**: 浅色主题

## 🔧 故障排除

### 常见问题

1. **模块导入错误**
   - 检查requirements.txt是否包含所有依赖项
   - 确保版本号兼容

2. **文件路径错误**
   - 已修复：模型文件路径指向`models/`目录
   - 已修复：数据文件路径指向`data/`目录

3. **内存不足**
   - Streamlit Share有内存限制
   - 如果模型文件过大，考虑模型压缩

4. **部署失败**
   - 检查GitHub仓库是否公开
   - 确认所有文件都已推送到仓库

## 📱 使用建议

### 分享您的应用
部署成功后，您可以：
- 将URL分享给同事和合作伙伴
- 嵌入到网站或文档中
- 用于演示和教学

### 更新应用
- 直接推送代码到GitHub仓库
- Streamlit Share会自动重新部署
- 通常在几分钟内完成更新

## 🎉 完成！

恭喜！您的肺癌生存分析系统现在已经部署到云端，任何人都可以通过URL访问使用。

---

**系统功能预览**：
- 📊 数据上传和预处理
- 🔬 PFS/OS生存分析
- 📈 生存曲线可视化
- 🎯 风险评分计算
- 🏥 患者管理系统
- 📋 个性化报告生成 