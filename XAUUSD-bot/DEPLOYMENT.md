# XAUUSD Trading Bot - Render Deployment

## Quick Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## Manual Deployment Steps

1. **Fork/Use this Repository**
   - Repository: https://github.com/DevGohil411/Trading-bot

2. **Sign up on Render**
   - Go to https://render.com
   - Sign up with GitHub

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `DevGohil411/Trading-bot`
   - Configure:
     - **Name:** xauusd-trading-bot
     - **Region:** Oregon (US West)
     - **Root Directory:** `XAUUSD-bot`
     - **Runtime:** Python 3
     - **Build Command:** `pip install --no-cache-dir -r requirements.txt`
     - **Start Command:** `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

4. **Add Environment Variables** (if needed)
   - Go to Environment tab
   - Add any API keys or secrets from your `.env` file

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete

## API Endpoints

- Health Check: `GET /api/account`
- Bot Status: `GET /api/bot/status`
- Start Bot: `POST /api/bot/start`
- Stop Bot: `POST /api/bot/stop`

## Frontend URL

Frontend deployed at: https://trading-bot-main-iy2iaveyl-dev-gohils-projects.vercel.app

## Notes

- MetaTrader5 will not work in cloud deployment (Windows only)
- Bot can be controlled via API endpoints
- Free tier on Render spins down after 15 minutes of inactivity
