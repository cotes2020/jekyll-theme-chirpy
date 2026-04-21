/**
 * BotContext — dispatchSlashCommand 등에 전달되는 서비스 묶음.
 * main.ts의 buildCtx() 반환값과 일치해야 함.
 */
import type { Client } from 'discord.js';
import type { GameDataService } from '../../services/gamedata';
import type { EnhancementService } from '../../services/enhancement';
import type { StockService } from '../../services/stock';
import type { RaidService } from '../../services/raid';
import type { CharacterService } from '../../services/character-service';
import type { MemoryService } from '../../services/memory-service';
import type { ScheduleService } from '../../services/schedule-service';
import type { MoodService } from '../../services/mood-service';
import type { GenerativeTextClient } from 'karmolab-ai/node';

export interface BotContext {
  client: Client;
  gameData: GameDataService;
  enhancement: EnhancementService;
  stock: StockService;
  raid: RaidService;
  characterService: CharacterService | null;
  getMemory: ((slug: string) => MemoryService) | null;
  getSchedule: ((slug: string) => ScheduleService) | null;
  getMood: ((slug: string) => MoodService) | null;
  getImageAttachment: (imageRelativePath: string) => { file: string; name: string } | null;
  isAdmin: (userId: unknown) => boolean;
  generativeText: GenerativeTextClient | null;
  cursorState: { inFlight: boolean };
}
