#!/bin/bash

echo "🚀 DigitalOcean Trading Signals App Setup"
echo "=========================================="

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Create .do directory if it doesn't exist
mkdir -p .do

# Update app.yaml with correct GitHub username
cat > .do/app.yaml << EOF
name: trading-signals-app
services:
- name: web
  source_dir: /
  github:
    repo: $GITHUB_USERNAME/trading-signals-app
    branch: main
  run_command: streamlit run app.py --server.port \$PORT --server.address 0.0.0.0
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  routes:
  - path: /
  envs:
  - key: PYTHON_VERSION
    value: "3.9"
EOF

# Create Dockerfile for alternative deployment
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# Create .dockerignore
cat > .dockerignore << EOF
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
.env
.venv
venv/
ENV/
env/
.idea/
.vscode/
*.swp
*.swo
*~
EOF

# Create deployment instructions
cat > DEPLOYMENT.md << EOF
# 🚀 DigitalOcean Deployment Guide

## Quick Deploy (15 minutes)

### Step 1: GitHub Repository
1. Create new repository: \`trading-signals-app\`
2. Make it public
3. Push your code:
\`\`\`bash
git remote add origin https://github.com/$GITHUB_USERNAME/trading-signals-app.git
git branch -M main
git push -u origin main
\`\`\`

### Step 2: DigitalOcean App Platform
1. Go to: https://cloud.digitalocean.com/apps
2. Click "Create App"
3. Connect GitHub account
4. Select repository: \`$GITHUB_USERNAME/trading-signals-app\`
5. Configure:
   - **Plan**: Basic (\$5/month or FREE with credits)
   - **Region**: Choose closest to you
   - **Branch**: main
6. Click "Create Resources"

### Step 3: Wait for Deployment
- Build time: 3-5 minutes
- You'll get a live URL like: \`https://trading-signals-xxxxx.ondigitalocean.app\`

## Features You Get:
✅ 99.9% uptime SLA
✅ Global CDN & SSL certificates  
✅ Auto-scaling & zero-downtime deploys
✅ Professional monitoring dashboard
✅ \$5/month (often FREE with \$200 credits)

## Development Workflow:
\`\`\`bash
# Make changes locally
git add .
git commit -m "Added new feature"
git push origin main
# DigitalOcean automatically rebuilds & deploys!
\`\`\`

## Custom Domain (Optional):
1. Add domain in DigitalOcean dashboard
2. Point CNAME to your app URL
3. SSL certificate auto-provisioned

## Monitoring:
- Built-in metrics in DigitalOcean dashboard
- Set up alerts for performance issues
- Monitor costs and usage

## Support:
- DigitalOcean documentation: https://docs.digitalocean.com/products/app-platform/
- Community forum: https://www.digitalocean.com/community/

---
**Your trading signals app will be competing with commercial platforms!** 🎯
EOF

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create GitHub repo: trading-signals-app"
echo "2. Push code to GitHub"
echo "3. Deploy on DigitalOcean App Platform"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions"
echo ""
echo "🎯 Your professional trading platform will be live in 15 minutes!" 