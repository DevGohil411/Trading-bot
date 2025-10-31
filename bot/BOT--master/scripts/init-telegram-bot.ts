import TelegramBotService from '../lib/telegram-bot';

// Initialize the Telegram bot service
async function initializeTelegramBot() {
  try {
    console.log('🤖 Initializing Telegram Bot...');
    
    // Get the bot service instance
    const botService = TelegramBotService.getInstance();
    
    // Send a startup notification
    await botService.sendBotStatusNotification('Bot Initialized', 'SniprX Trading Bot is now online and ready to receive commands.');
    
    console.log('✅ Telegram Bot initialized successfully');
    console.log('📱 Bot is ready to receive commands and send notifications');
    console.log('🔗 Users can start the bot with /start command');
    
  } catch (error) {
    console.error('❌ Failed to initialize Telegram Bot:', error);
  }
}

// Run the initialization
if (require.main === module) {
  initializeTelegramBot();
}

export default initializeTelegramBot; 