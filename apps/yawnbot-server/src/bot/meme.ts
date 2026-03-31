// @ts-nocheck
import path from 'path';
import fs from 'fs';
import { EmbedBuilder } from 'discord.js';
import type { Message } from 'discord.js';
import { memeImgDir } from '../paths';

const MEME_DIR = memeImgDir();

export async function handleMeme(message: Message): Promise<boolean> {
    if (message.author.bot || message.content.startsWith('!') || message.content.startsWith('/')) return false;
    const query = message.content.trim().toLowerCase();
    if (!query) return false;

    try {
        if (!fs.existsSync(MEME_DIR)) return false;
        const files = fs.readdirSync(MEME_DIR);
        const match = files.find(f => path.parse(f).name.toLowerCase() === query);
        if (match) {
            const embed = new EmbedBuilder()
                .setTitle(`🖼️ ${query}`)
                .setImage(`attachment://${match}`)
                .setColor(0xffd700)
                .setFooter({ text: `Requested by ${message.author.username}`, iconURL: message.author.displayAvatarURL() });
            await message.channel.send({ embeds: [embed], files: [path.join(MEME_DIR, match)] });
            return true;
        }
    } catch (e) {
        /* ignore */
    }
    return false;
}
