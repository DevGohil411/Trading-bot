# Telegram Bot Integration Guide

## Overview

This document explains how to set up and use the Telegram bot integration for SniprX Trading Bot. The bot provides real-time notifications and allows users to check trading status, account balance, and recent trades through Telegram.

## Features

- 🔐 **Login/Logout Notifications** - Get notified when users log in or out
- 📊 **Trade Notifications** - Real-time alerts for new trades, profits, and losses
- 🤖 **Bot Status Updates** - Notifications when the trading bot starts/stops
- ⚠️ **Error Alerts** - Get notified of any system errors
- 📈 **Trading Commands** - Check status, balance, trades, and summary via Telegram

## Setup Instructions

### 1. Bot Token Configuration

The bot token is already configured in the system:
```
Token: 8435934615:AAEDS34JKRkjBcPRrdtOrd5i3NKs4OLFzJY
```

### 2. Environment Variables

Add these to your `.env.local` file:
```bash
TELEGRAM_BOT_TOKEN=8435934615:AAEDS34JKRkjBcPRrdtOrd5i3NKs4OLFzJY
TELEGRAM_BOT_USERNAME=SniprXBot
```

### 3. Starting the Bot

The bot will automatically start when your Next.js application starts. The bot service is initialized in `lib/telegram-bot.ts`.

### 4. Authorizing Users

To authorize users to receive notifications:

1. Users need to start a conversation with the bot by sending `/start`
2. The bot will show their Chat ID
3. Add the Chat ID to the authorized users list in the Telegram Bot Manager (Settings > Alerts)

## Bot Commands

### Available Commands

- `/start` - Initialize the bot and get welcome message
- `/help` - Show all available commands
- `/status` - Get current trading status and bot status
- `/balance` - Check MT5 account balance and equity
- `/trades` - View recent open and closed trades
- `/summary` - Get trading summary and performance stats
- `/notifications` - Toggle trading notifications

### Command Examples

```
/status
```
Response:
```
📊 Trading Status

🔄 Connection Status: 🟢 Connected
💰 Balance: $10,000.00
📈 Equity: $10,250.00
📊 Open Trades: 2
✅ Closed Trades: 15
🎯 Win Rate: 73.3%
```

```
/trades
```
Response:
```
📈 Recent Trades

🔄 Open Trades (2):
🟢 EURUSD BUY 0.1L - +$25.00
🔴 GBPUSD SELL 0.1L - -$15.00

✅ Recent Closed Trades (5):
🟢 EURUSD BUY 0.1L - +$45.00
🔴 GBPUSD SELL 0.1L - -$20.00
🟢 USDJPY BUY 0.1L - +$30.00
```

## Notification Types

### 1. Login/Logout Notifications
```
🔐 User Login

👤 User: trader@example.com
🆔 ID: 1
⏰ Time: 12/19/2024, 2:30:45 PM
```

### 2. Trade Notifications
```
🟢 TRADE

📊 Symbol: EURUSD
📈 Type: BUY
📊 Volume: 0.1L
💰 Profit: +$25.00
⏰ Time: 12/19/2024, 2:30:45 PM
```

### 3. Bot Status Updates
```
🟢 Bot Status Update

🔄 Status: Bot Started
📝 Details: Strategy: mmxm, Killzone: Enabled
⏰ Time: 12/19/2024, 2:30:45 PM
```

### 4. Error Alerts
```
⚠️ Error Alert

❌ Error: Failed to connect to MT5
⏰ Time: 12/19/2024, 2:30:45 PM
```

## Configuration

### Telegram Bot Manager

Access the Telegram Bot Manager through:
1. Go to Settings > Alerts
2. Configure notification preferences
3. Manage authorized users
4. Test notifications

### Notification Settings

You can enable/disable specific notification types:
- ✅ Login/Logout Notifications
- ✅ Trade Notifications  
- ✅ Bot Status Updates
- ✅ Error Alerts

## API Endpoints

### Send Notification
```
POST /api/telegram/notify
Content-Type: application/json

{
  "type": "trade|login|logout|bot_status|error",
  "data": {
    // Notification specific data
  }
}
```

### Webhook (for receiving messages)
```
POST /api/telegram/webhook
```

## Troubleshooting

### Common Issues

1. **Bot not responding**
   - Check if the bot token is correct
   - Ensure the bot is running (check console logs)
   - Verify the bot hasn't been blocked by users

2. **Notifications not sending**
   - Check if users are authorized
   - Verify notification settings are enabled
   - Check network connectivity

3. **Commands not working**
   - Ensure users are authorized
   - Check if the bot has proper permissions
   - Verify the API endpoints are accessible

### Debug Mode

Enable debug logging by adding to your environment:
```bash
DEBUG=telegram-bot
```

## Security Considerations

1. **Token Security**
   - Never expose the bot token in client-side code
   - Use environment variables for sensitive data
   - Rotate tokens regularly

2. **User Authorization**
   - Only authorize trusted users
   - Implement proper user management
   - Monitor bot usage

3. **Rate Limiting**
   - Implement rate limiting for commands
   - Monitor for spam or abuse
   - Set appropriate limits

## Development

### Adding New Commands

1. Add command handler in `lib/telegram-bot.ts`
2. Update the `setupCommands()` method
3. Test the command thoroughly
4. Update documentation

### Adding New Notifications

1. Add notification method in `TelegramBotService`
2. Update the notification API endpoint
3. Integrate with existing components
4. Test the notification flow

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the console logs
3. Test with the `/help` command
4. Contact the development team

## Future Enhancements

- [ ] Webhook support for real-time updates
- [ ] Advanced notification scheduling
- [ ] Custom notification templates
- [ ] Multi-language support
- [ ] Analytics and usage tracking
- [ ] Integration with other messaging platforms 