import { NextApiRequest, NextApiResponse } from 'next';
import { TelegramBotService } from '@/lib/telegram-bot';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { type, data } = req.body;

    if (!type || !data) {
      return res.status(400).json({ message: 'Type and data are required' });
    }

    const botService = TelegramBotService.getInstance();

    switch (type) {
      case 'login':
        await botService.sendLoginNotification(data.userId, data.userEmail);
        break;
      case 'logout':
        await botService.sendLogoutNotification(data.userId, data.userEmail);
        break;
      case 'trade':
        await botService.sendTradeNotification(data.trade);
        break;
      case 'bot_status':
        await botService.sendBotStatusNotification(data.status, data.details);
        break;
      case 'error':
        await botService.sendErrorNotification(data.error);
        break;
      default:
        return res.status(400).json({ message: 'Invalid notification type' });
    }

    res.status(200).json({ success: true, message: 'Notification sent successfully' });
  } catch (error) {
    console.error('Telegram notification error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ message: 'Failed to send notification', error: errorMessage });
  }
} 