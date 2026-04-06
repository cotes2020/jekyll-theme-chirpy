import fs from 'fs';
import path from 'path';

/**
 * karmo-anime 카탈로그에 새 항목을 추가하는 스크립트
 * 
 * 사용법:
 *   1. 단일 항목 추가: node add-item.js "작품 이름" [이미지키]
 *   2. 파일로 여러 항목 추가: node add-item.js --file names.txt
 * 
 * 실행 위치: apps/karmolab/src/widgets/tierlist/scripts
 */

// 레포 루트 기준으로 경로 설정
const CATALOG_PATH = '../../../../data/tierlists/karmo-anime.json';

function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

async function addItem() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.error('사용법:');
        console.error('  단일 추가: node add-item.js "작품 이름" [이미지키]');
        console.error('  파일 추가: node add-item.js --file names.txt');
        process.exit(1);
    }

    let itemsToAdd = [];

    if (args[0] === '--file' && args[1]) {
        const filePath = path.resolve(args[1]);
        if (!fs.existsSync(filePath)) {
            console.error(`파일을 찾을 수 없습니다: ${filePath}`);
            process.exit(1);
        }
        const content = fs.readFileSync(filePath, 'utf-8');
        itemsToAdd = content.split(/\r?\n/)
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .map(name => ({ name, imageKey: null }));
    } else {
        itemsToAdd.push({ name: args[0], imageKey: args[1] || null });
    }

    if (itemsToAdd.length === 0) {
        console.warn('추가할 항목이 없습니다.');
        return;
    }

    try {
        const fullPath = path.resolve(import.meta.dirname, CATALOG_PATH);
        if (!fs.existsSync(fullPath)) {
            console.error(`카탈로그 파일을 찾을 수 없습니다: ${fullPath}`);
            process.exit(1);
        }

        const data = JSON.parse(fs.readFileSync(fullPath, 'utf-8'));
        if (!data.items) data.items = {};

        let addedCount = 0;
        for (const item of itemsToAdd) {
            const id = 'ti-' + uid();
            
            // 중복 이름 체크
            const isDuplicate = Object.values(data.items).some(existing => existing.name === item.name);
            if (isDuplicate) {
                console.warn(`주의: "${item.name}" 이름이 이미 카탈로그에 존재하여 건너뜁니다.`);
                continue;
            }

            data.items[id] = {
                id: id,
                name: item.name
            };
            
            if (item.imageKey) {
                data.items[id].imageKey = item.imageKey;
            }
            addedCount++;
        }

        if (addedCount > 0) {
            // ID 순으로 정렬하여 diff 최소화
            const sortedItems = {};
            Object.keys(data.items).sort().forEach(key => {
                sortedItems[key] = data.items[key];
            });
            data.items = sortedItems;

            fs.writeFileSync(fullPath, JSON.stringify(data, null, 2), 'utf-8');
            console.log(`성공: 총 ${addedCount}개의 항목이 카탈로그에 추가되었습니다.`);
        } else {
            console.log('추가된 항목이 없습니다.');
        }
    } catch (error) {
        console.error('오류 발생:', error.message);
        process.exit(1);
    }
}

addItem();
