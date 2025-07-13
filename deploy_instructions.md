# 🚀 Deployment Instructions for Streamlit Community Cloud

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name your repository (e.g., `ai-trading-signals`)
5. Make it **Public** (required for Streamlit Community Cloud)
6. **Don't** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 2: Connect Your Local Repository to GitHub

After creating the GitHub repository, you'll see instructions. Run these commands in your terminal:

```bash
# Add the remote repository (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository from the dropdown
5. Set the main file path to: `main.py`
6. Click "Deploy!"

## Step 4: Configure Environment Variables (Optional)

If you want to use your own Alpha Vantage API key:

1. In your Streamlit app settings, add an environment variable:
   - Name: `ALPHA_VANTAGE_KEY`
   - Value: Your API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

2. Update your `main.py` to use the environment variable:
   ```python
   import os
   ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "FAA6B54PAH8QH33Y")
   ```

## Troubleshooting

- **Repository not found**: Make sure your repository is public
- **Deployment fails**: Check that `main.py` is in the root directory
- **API errors**: Verify your Alpha Vantage API key is valid
- **Dependencies missing**: Ensure `requirements.txt` is in the root directory

## Quick Commands

```bash
# Check current status
git status

# Add changes
git add .

# Commit changes
git commit -m "Update app"

# Push to GitHub
git push origin main
```

Your app will be available at: `https://YOUR_APP_NAME.streamlit.app` 