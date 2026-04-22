import fs from 'fs';
import path from 'path';
import { enhancementImgDir } from '../paths';

const ENHANCEMENT_DIR = enhancementImgDir();

export function getImageAttachment(imageRelativePath: string | null | undefined): { file: string; name: string } | null {
  if (!imageRelativePath) return null;
  const fullPath = path.join(ENHANCEMENT_DIR, imageRelativePath);
  if (fs.existsSync(fullPath)) {
    const name = path.basename(fullPath);
    return { file: fullPath, name };
  }
  return null;
}

