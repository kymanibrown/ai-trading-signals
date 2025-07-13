#!/bin/bash

# Replace YOUR_USERNAME with your actual GitHub username
# Replace REPO_NAME with your repository name (e.g., ai-trading-signals)

echo "🚀 Setting up GitHub repository connection..."

# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Set the main branch
git branch -M main

# Push to GitHub
git push -u origin main

echo "✅ Code pushed to GitHub successfully!"
echo "🌐 Your repository is now available at: https://github.com/YOUR_USERNAME/REPO_NAME"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://share.streamlit.io"
echo "2. Sign in with your GitHub account"
echo "3. Click 'New app'"
echo "4. Select your repository"
echo "5. Set main file path to: main.py"
echo "6. Click 'Deploy!'" 