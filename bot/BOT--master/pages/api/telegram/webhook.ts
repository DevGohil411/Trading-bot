import { NextApiRequest, NextApiResponse } from 'next';
import { TelegramBotService } from '@/lib/telegram-bot';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { message } = req.body;

    if (!message) {
      return res.status(400).json({ message: 'No message received' });
    }

    // Handle the message through the Telegram bot service
    const botService = TelegramBotService.getInstance();
    
    // The bot service will handle the message automatically
    // This webhook is mainly for receiving updates from Telegram

    res.status(200).json({ success: true });
  } catch (error) {
    console.error('Telegram webhook error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ message: 'Internal server error', error: errorMessage });
  }
} 