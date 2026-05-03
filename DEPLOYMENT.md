# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub account with repository pushed (✅ Already done - `Naveen-567/SpectrTwin`)
- Streamlit Cloud account

## Step-by-Step Deployment

### 1. Go to Streamlit Cloud
- Visit https://streamlit.io/cloud
- Sign in with GitHub

### 2. Create New App
- Click "New app"
- Repository: `Naveen-567/SpectrTwin`
- Branch: `main`
- Main file path: `Home.py`
- Click "Deploy"

### 3. Set Up Secrets
While the app deploys, set up your Groq API key:

1. Go to app settings (gear icon) → **Secrets**
2. Paste the following into the secrets editor:
```toml
GROQ_API_KEY = "your_actual_groq_api_key_here"
```
3. Click "Save"

**To get your Groq API Key:**
- Visit https://console.groq.com/keys
- Create a new API key
- Copy and paste it into Streamlit secrets

### 4. Wait for Deployment
- The app will rebuild automatically
- Check the deployment logs for errors
- Once done, you'll see "Your app is ready!"

## Troubleshooting

### Error: "metadata-generation-failed"
**Cause:** Package build failure on Streamlit Cloud  
**Solution:**
- This is now fixed in the latest version
- Go to your app settings → Reboot app
- Or make a small commit and push to trigger redeployment

### Error: "ModuleNotFoundError: No module named 'groq'"
**Solution:**
- This is expected if groq fails to install
- The app will work without the chatbot feature
- To force reinstall: Go to app settings → Advanced settings → Reboot app

### Error: "Your app is having trouble"
**Solution:**
1. Check deployment logs (gear icon → Manage app → Logs)
2. Look for the actual error message
3. The most common issues are:
   - Missing dependencies in `requirements.txt`
   - Incorrect API key format
   - Module import issues

### Chatbot not working
**Solution:**
- Verify `GROQ_API_KEY` is correctly set in Secrets
- Check that the key is active on Groq console
- The app functions without the chatbot if the key is missing

## Important Notes

⚠️ **Do NOT commit:**
- `Api.txt` with real API keys (already in .gitignore)
- `.streamlit/secrets.toml` (local only)
- `__pycache__/` directories

✅ **Do commit:**
- `requirements.txt` (all dependencies)
- `.streamlit/config.toml` (public settings)
- `.streamlit/secrets.toml.example` (template only)

## App URL
Once deployed, your app will be at:
```
https://spectrtwin-[random-id].streamlit.app
```

## Support
If you encounter issues:
1. Check the deployment logs
2. Review the Streamlit Cloud documentation
3. Check if all packages in `requirements.txt` are compatible

---

**Last Updated:** May 4, 2026
