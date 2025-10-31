import { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Only return the bot username, never the token
    const botUsername = process.env.TELEGRAM_BOT_USERNAME || 'SniprXBot';
    
    res.status(200).json({ 
      success: true, 
      botInfo: {
        username: botUsername.startsWith('@') ? botUsername : `@${botUsername}`
      }
    });
  } catch (error) {
    console.error('Error fetching bot info:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    res.status(500).json({ message: 'Failed to fetch bot info', error: errorMessage });
  }
} 