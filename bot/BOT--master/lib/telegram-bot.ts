import type { NextRequest } from 'next/server';

// Initialize the bot with token from environment variables
const token = process.env.TELEGRAM_BOT_TOKEN;
let bot: TelegramBot | null = null;

// Only create bot instance if token is available and we're on the server side
if (token && typeof window === 'undefined') {
  try {
    bot = new TelegramBot(token, { 
      polling: true
    });
    
    // Handle polling errors
    bot.on('polling_error', (error) => {
      console.error('Telegram bot polling error:', error.message);
      // Don't throw error, just log it
    });
    
    console.log('✅ Telegram bot initialized successfully');
  } catch (error) {
    console.error('❌ Failed to initialize Telegram bot:', error);
    bot = null;
  }
} else {
  console.log('⚠️ Telegram bot not initialized - client side or no token');
}

// Store authorized users (in production, use a database)
const authorizedUsers = new Set<number>();

// Bot commands and handlers
export class TelegramBotService {
  private static instance: TelegramBotService;
  private bot: TelegramBot | null;

  private constructor() {
    this.bot = bot;
    if (this.bot) {
      this.setupCommands();
    } else {
      console.warn('Telegram bot not initialized - no token provided or client side');
    }
  }

  public static getInstance(): TelegramBotService {
    if (!TelegramBotService.instance) {
      TelegramBotService.instance = new TelegramBotService();
    }
    return TelegramBotService.instance;
  }

  private setupCommands() {
    if (!this.bot) return;

    // Start command
    this.bot.onText(/\/start/, (msg) => {
      const chatId = msg.chat.id;
      const welcomeMessage = `
🤖 Welcome to SniprX Trading Bot!

I'm here to keep you updated on your trading activities. Here are the available commands:

📊 /status - Get current trading status
💰 /balance - Check account balance
📈 /trades - View recent trades
🔔 /notifications - Toggle notifications
❓ /help - Show this help message

To receive notifications, please contact the admin to authorize your chat ID: ${chatId}
      `;
      this.bot?.sendMessage(chatId, welcomeMessage);
    });

    // Help command
    this.bot.onText(/\/help/, (msg) => {
      const chatId = msg.chat.id;
      const helpMessage = `
🤖 SniprX Trading Bot Commands:

📊 /status - Get current trading status and bot status
💰 /balance - Check your MT5 account balance and equity
📈 /trades - View recent open and closed trades
🔔 /notifications - Toggle trading notifications on/off
📋 /summary - Get trading summary and performance stats
❓ /help - Show this help message

For support, contact your trading administrator.
      `;
      this.bot?.sendMessage(chatId, helpMessage);
    });

    // Status command
    this.bot.onText(/\/status/, async (msg) => {
      const chatId = msg.chat.id;
      if (!this.isAuthorized(chatId)) {
        this.bot?.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
        return;
      }

      try {
        const status = await this.getTradingStatus();
        this.bot?.sendMessage(chatId, status, { parse_mode: 'HTML' });
      } catch (error) {
        this.bot?.sendMessage(chatId, "❌ Error fetching status. Please try again later.");
      }
    });

    // Balance command
    this.bot.onText(/\/balance/, async (msg) => {
      const chatId = msg.chat.id;
      if (!this.isAuthorized(chatId)) {
        this.bot?.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
        return;
      }

      try {
        const balance = await this.getAccountBalance();
        this.bot?.sendMessage(chatId, balance, { parse_mode: 'HTML' });
      } catch (error) {
        this.bot?.sendMessage(chatId, "❌ Error fetching balance. Please try again later.");
      }
    });

    // Trades command
    this.bot.onText(/\/trades/, async (msg) => {
      const chatId = msg.chat.id;
      if (!this.isAuthorized(chatId)) {
        this.bot?.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
        return;
      }

      try {
        const trades = await this.getRecentTrades();
        this.bot?.sendMessage(chatId, trades, { parse_mode: 'HTML' });
      } catch (error) {
        this.bot?.sendMessage(chatId, "❌ Error fetching trades. Please try again later.");
      }
    });

    // Summary command
    this.bot.onText(/\/summary/, async (msg) => {
      const chatId = msg.chat.id;
      if (!this.isAuthorized(chatId)) {
        this.bot?.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
        return;
      }

      try {
        const summary = await this.getTradingSummary();
        this.bot?.sendMessage(chatId, summary, { parse_mode: 'HTML' });
      } catch (error) {
        this.bot?.sendMessage(chatId, "❌ Error fetching summary. Please try again later.");
      }
    });

    // Notifications toggle
    this.bot.onText(/\/notifications/, (msg) => {
      const chatId = msg.chat.id;
      if (!this.isAuthorized(chatId)) {
        this.bot?.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
        return;
      }

      // Toggle notifications (in production, store in database)
      const message = "🔔 Notifications are currently enabled. Use /notifications_off to disable.";
      this.bot?.sendMessage(chatId, message);
    });
  }

  private isAuthorized(chatId: number): boolean {
    // In production, check against database
    return authorizedUsers.has(chatId) || chatId === 123456789; // Replace with actual admin chat ID
  }

  private async getTradingStatus(): Promise<string> {
    try {
      const [accountsRes, tradesRes] = await Promise.all([
        fetch(`/api/mt5/accounts`),
        fetch(`/api/mt5/trades`)
      ]);

      const accounts = await accountsRes.json();
      const trades = await tradesRes.json();

      const account = accounts?.[0] || {};
      const openTrades = trades?.open_trades || [];
      const closedTrades = trades?.closed_trades || [];

      return `
📊 <b>Trading Status</b>

🔄 <b>Connection Status:</b> ${account.connected ? '🟢 Connected' : '🔴 Disconnected'}
💰 <b>Balance:</b> $${account.balance?.toFixed(2) || '0.00'}
📈 <b>Equity:</b> $${account.equity?.toFixed(2) || '0.00'}
📊 <b>Open Trades:</b> ${openTrades.length}
✅ <b>Closed Trades:</b> ${closedTrades.length}
🎯 <b>Win Rate:</b> ${account.winRate?.toFixed(1) || '0'}%
      `;
    } catch (error) {
      return "❌ Unable to fetch trading status";
    }
  }

  private async getAccountBalance(): Promise<string> {
    try {
      const res = await fetch(`/api/mt5/accounts`);
      const accounts = await res.json();
      const account = accounts?.[0] || {};

      return `
💰 <b>Account Balance</b>

🏦 <b>Balance:</b> $${account.balance?.toFixed(2) || '0.00'}
📊 <b>Equity:</b> $${account.equity?.toFixed(2) || '0.00'}
💼 <b>Margin:</b> $${account.margin?.toFixed(2) || '0.00'}
🆓 <b>Free Margin:</b> $${account.freeMargin?.toFixed(2) || '0.00'}
📈 <b>Margin Level:</b> ${account.marginLevel?.toFixed(1) || '0'}%
      `;
    } catch (error) {
      return "❌ Unable to fetch account balance";
    }
  }

  private async getRecentTrades(): Promise<string> {
    try {
      const res = await fetch(`/api/mt5/trades`);
      const trades = await res.json();
      
      const openTrades = trades?.open_trades || [];
      const closedTrades = trades?.closed_trades || [];

      let message = `📈 <b>Recent Trades</b>\n\n`;

      if (openTrades.length > 0) {
        message += `🔄 <b>Open Trades (${openTrades.length}):</b>\n`;
        openTrades.slice(0, 5).forEach((trade: any) => {
          const profit = trade.profit >= 0 ? `+$${trade.profit.toFixed(2)}` : `-$${Math.abs(trade.profit).toFixed(2)}`;
          const emoji = trade.profit >= 0 ? '🟢' : '🔴';
          message += `${emoji} ${trade.symbol} ${trade.type} ${trade.volume}L - ${profit}\n`;
        });
      }

      if (closedTrades.length > 0) {
        message += `\n✅ <b>Recent Closed Trades (${closedTrades.length}):</b>\n`;
        closedTrades.slice(0, 5).forEach((trade: any) => {
          const profit = trade.profit >= 0 ? `+$${trade.profit.toFixed(2)}` : `-$${Math.abs(trade.profit).toFixed(2)}`;
          const emoji = trade.profit >= 0 ? '🟢' : '🔴';
          message += `${emoji} ${trade.symbol} ${trade.type} ${trade.volume}L - ${profit}\n`;
        });
      }

      if (openTrades.length === 0 && closedTrades.length === 0) {
        message += "No trades found.";
      }

      return message;
    } catch (error) {
      return "❌ Unable to fetch trades";
    }
  }

  private async getTradingSummary(): Promise<string> {
    try {
      const [accountsRes, tradesRes] = await Promise.all([
        fetch(`/api/mt5/accounts`),
        fetch(`/api/mt5/trades`)
      ]);

      const accounts = await accountsRes.json();
      const trades = await tradesRes.json();

      const account = accounts?.[0] || {};
      const closedTrades = trades?.closed_trades || [];
      const openTrades = trades?.open_trades || [];

      const wins = closedTrades.filter((t: any) => t.profit > 0);
      const losses = closedTrades.filter((t: any) => t.profit < 0);
      const totalProfit = closedTrades.reduce((sum: number, t: any) => sum + t.profit, 0);
      const avgWin = wins.length > 0 ? wins.reduce((sum: number, t: any) => sum + t.profit, 0) / wins.length : 0;
      const avgLoss = losses.length > 0 ? losses.reduce((sum: number, t: any) => sum + t.profit, 0) / losses.length : 0;

      return `
📋 <b>Trading Summary</b>

💰 <b>Total Profit:</b> ${totalProfit >= 0 ? '+' : ''}$${totalProfit.toFixed(2)}
📊 <b>Total Trades:</b> ${closedTrades.length}
✅ <b>Winning Trades:</b> ${wins.length}
❌ <b>Losing Trades:</b> ${losses.length}
🎯 <b>Win Rate:</b> ${closedTrades.length > 0 ? ((wins.length / closedTrades.length) * 100).toFixed(1) : '0'}%
📈 <b>Average Win:</b> $${avgWin.toFixed(2)}
📉 <b>Average Loss:</b> $${avgLoss.toFixed(2)}
🔄 <b>Open Trades:</b> ${openTrades.length}
      `;
    } catch (error) {
      return "❌ Unable to fetch trading summary";
    }
  }

  // Public methods for sending notifications
  public async sendLoginNotification(userId: string, userEmail: string): Promise<void> {
    const message = `
🔐 <b>User Login</b>

👤 <b>User:</b> ${userEmail}
🆔 <b>ID:</b> ${userId}
⏰ <b>Time:</b> ${new Date().toLocaleString()}
    `;
    await this.sendToAllAuthorizedUsers(message);
  }

  public async sendLogoutNotification(userId: string, userEmail: string): Promise<void> {
    const message = `
🚪 <b>User Logout</b>

👤 <b>User:</b> ${userEmail}
🆔 <b>ID:</b> ${userId}
⏰ <b>Time:</b> ${new Date().toLocaleString()}
    `;
    await this.sendToAllAuthorizedUsers(message);
  }

  public async sendTradeNotification(trade: any): Promise<void> {
    const profit = trade.profit >= 0 ? `+$${trade.profit.toFixed(2)}` : `-$${Math.abs(trade.profit).toFixed(2)}`;
    const emoji = trade.profit >= 0 ? '🟢' : '🔴';
    const action = trade.action || 'TRADE';
    
    const message = `
${emoji} <b>${action}</b>

📊 <b>Symbol:</b> ${trade.symbol}
📈 <b>Type:</b> ${trade.type}
📊 <b>Volume:</b> ${trade.volume}L
💰 <b>Profit:</b> ${profit}
⏰ <b>Time:</b> ${new Date().toLocaleString()}
    `;
    await this.sendToAllAuthorizedUsers(message);
  }

  public async sendBotStatusNotification(status: string, details?: string): Promise<void> {
    const emoji = status.toLowerCase().includes('start') ? '🟢' : '🔴';
    const message = `
${emoji} <b>Bot Status Update</b>

🔄 <b>Status:</b> ${status}
${details ? `📝 <b>Details:</b> ${details}\n` : ''}
⏰ <b>Time:</b> ${new Date().toLocaleString()}
    `;
    await this.sendToAllAuthorizedUsers(message);
  }

  public async sendErrorNotification(error: string): Promise<void> {
    const message = `
⚠️ <b>Error Alert</b>

❌ <b>Error:</b> ${error}
⏰ <b>Time:</b> ${new Date().toLocaleString()}
    `;
    await this.sendToAllAuthorizedUsers(message);
  }

  private async sendToAllAuthorizedUsers(message: string): Promise<void> {
    if (!this.bot) {
      console.log('Bot not initialized - cannot send notification');
      return;
    }

    // In production, get authorized users from database
    const authorizedChatIds = Array.from(authorizedUsers);
    
    if (authorizedChatIds.length === 0) {
      console.log('No authorized users to send notification to');
      return;
    }
    
    for (const chatId of authorizedChatIds) {
      try {
        await this.bot.sendMessage(chatId, message, { parse_mode: 'HTML' });
        console.log(`Notification sent successfully to chat ${chatId}`);
      } catch (error) {
        console.error(`Failed to send message to chat ${chatId}:`, error);
      }
    }
  }

  // Method to authorize a user
  public authorizeUser(chatId: number): void {
    authorizedUsers.add(chatId);
  }

  // Method to deauthorize a user
  public deauthorizeUser(chatId: number): void {
    authorizedUsers.delete(chatId);
  }
}

export default TelegramBotService.getInstance(); 