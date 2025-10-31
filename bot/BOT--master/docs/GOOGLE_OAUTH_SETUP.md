# Google OAuth 2.0 Authentication Setup

This document describes the Google OAuth 2.0 authentication implementation for the SniprX trading platform.

## Overview

The application uses Google OAuth 2.0 for user authentication with the following components:

- **Backend**: Express.js server with Passport.js for OAuth handling
- **Frontend**: Next.js application with authentication hooks
- **Session Management**: Express-session for server-side session storage

## Authentication Flow

1. User clicks "Continue with Google" button on login page
2. User is redirected to Google OAuth consent screen
3. After successful authentication, Google redirects to `/auth/google/callback`
4. User is redirected to `/dashboard` upon successful login
5. Session is maintained using express-session

## API Endpoints

### Authentication Routes

- `GET /auth/google` - Initiates Google OAuth flow
- `GET /auth/google/callback` - Google OAuth callback handler
- `GET /auth/status` - Check current authentication status
- `GET /logout` - Logout user and destroy session

### Protected API Routes

All API routes are protected and require authentication:

- `GET /api/account` - Get account data
- `GET /api/mt5/accounts` - Get MT5 accounts
- `GET /api/mt5/trades` - Get MT5 trades
- `GET /api/bot/settings` - Get bot settings
- `POST /api/bot/settings` - Update bot settings
- `GET /api/bot/status` - Get bot status
- `POST /api/bot/start` - Start trading bot
- `POST /api/bot/stop` - Stop trading bot
- `GET /api/telegram/bot-info` - Get Telegram bot info
- `POST /api/telegram/notify` - Send Telegram notifications

## Configuration

### Google OAuth Credentials

Set the following environment variables in `server/.env`:

- `GOOGLE_CLIENT_ID=<your-client-id>`
- `GOOGLE_CLIENT_SECRET=<your-client-secret>`
- `SESSION_SECRET=<a-long-random-secret>`

Ensure your Google Cloud OAuth consent screen and credentials allow the callback:
- **Callback URL**: `http://localhost:8000/auth/google/callback`

### Environment Variables

Create a `.env` file in the `server` directory with:

```env
PORT=8000
SESSION_SECRET=your-super-secret-session-key-change-in-production
NODE_ENV=development
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_BOT_USERNAME=your-telegram-bot-username
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

## Frontend Integration

### Login

Frontend routes to backend auth via rewrites configured in `next.config.mjs`. Hitting `/auth/google` in the browser will proxy to the backend.

### Authentication Hook

The `useAuth` hook (`hooks/use-auth.tsx`) manages authentication state and provides:

- `isAuthenticated`: Boolean indicating if user is logged in
- `user`: User object with profile information
- `logout()`: Function to logout user
- `loading`: Boolean indicating if auth check is in progress

### Protected Routes

The `ProtectedRoute` component ensures users are authenticated before accessing protected pages.

## Security Considerations

1. **Session Secret**: Use a strong, unique session secret in production
2. **HTTPS**: Enable HTTPS in production for secure cookie transmission
3. **CORS**: Configure CORS properly for your domain
4. **Environment Variables**: Store sensitive data in environment variables
5. **Session Storage**: Consider using Redis or database for session storage in production

## Development Setup

1. Start the backend server:
   ```bash
   cd server
   npm install
   npm start
   ```

2. Start the frontend application:
   ```bash
   npm install
   npm run dev
   ```

3. Access the application at `http://localhost:3000`

## Testing

1. Navigate to `http://localhost:3000/login`
2. Click "Continue with Google" (or visit `/auth/google`)
3. Complete Google OAuth flow
4. Verify redirect to dashboard
5. Test logout functionality

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure CORS is properly configured for your frontend domain
2. **Session Not Persisting**: Check session configuration and cookie settings
3. **OAuth Errors**: Verify Google OAuth credentials and callback URL
4. **Redirect Issues**: Ensure callback URL matches Google OAuth configuration

### Debug Endpoints

- `GET /health` - Server health check
- `GET /auth/status` - Authentication status check 