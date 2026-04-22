/**
 * StockService — C# StockService → Node.js 이식
 * 5개 종목 가격 변동 시뮬레이션 + 매수/매도/포트폴리오
 */
import { formatMoney, GameDataService } from './gamedata';

const INITIAL_STOCKS: Record<
  string,
  { symbol: string; name: string; price: number; desc: string; previousPrice?: number; priceHistory?: number[] }
> = {
  SAMSUNG: { symbol: 'SAMSUNG', name: '떡락전자', price: 70000, desc: '국민 주식. 파란불이 익숙하다.' },
  DOGE: { symbol: 'DOGE', name: '화성갈끄니까', price: 100, desc: '도지코인. 화성 갈 수 있을까?' },
  TESLA: { symbol: 'TESLA', name: '테슬라', price: 200000, desc: '전기차의 미래.' },
  APPLE: { symbol: 'APPLE', name: '사과', price: 150000, desc: '감성의 사과.' },
  BITCOIN: { symbol: 'BITCOIN', name: '비트코인', price: 50000000, desc: '디지털 금.' },
};

export class StockService {
  gameData: GameDataService;
  stocks: typeof INITIAL_STOCKS;
  private _interval: ReturnType<typeof setInterval> | null = null;

  constructor(gameData: GameDataService) {
    this.gameData = gameData;
    this.stocks = JSON.parse(JSON.stringify(INITIAL_STOCKS)) as typeof INITIAL_STOCKS;

    for (const sym in this.stocks) {
      const st = this.stocks[sym];
      st.previousPrice = st.price;
      st.priceHistory = [st.price];
    }
  }

  startMarket(): void {
    this.stopMarket();
    this._interval = setInterval(() => this.updatePrices(), 60 * 1000);
    console.log('[Stock] 주식 시장 시뮬레이션 시작 (1분 간격)');
  }

  stopMarket(): void {
    if (this._interval) {
      clearInterval(this._interval);
      this._interval = null;
    }
  }

  updatePrices(): void {
    for (const sym in this.stocks) {
      const stock = this.stocks[sym];
      stock.previousPrice = stock.price;
      const changePct = Math.random() < 0.05 ? Math.random() * 0.4 - 0.2 : Math.random() * 0.1 - 0.05;
      stock.price += Math.round(stock.price * changePct);
      if (stock.price < 1) stock.price = 1;
      if (!stock.priceHistory) stock.priceHistory = [];
      stock.priceHistory.push(stock.price);
      if (stock.priceHistory.length > 30) stock.priceHistory.shift();
    }
  }

  getStockList() {
    return Object.values(this.stocks).map((st) => {
      const diff = st.price - (st.previousPrice || st.price);
      const pct = st.previousPrice ? (diff / st.previousPrice) * 100 : 0;
      return { ...st, diff, pct };
    });
  }

  getChartUrl(symbol: string): string {
    const stock = this.stocks[symbol];
    if (!stock) return '';
    if (!stock.priceHistory || stock.priceHistory.length === 0) stock.priceHistory = [stock.price];

    const labels = stock.priceHistory.map((_, i) => `'${i + 1}m'`).join(',');
    const data = stock.priceHistory.join(',');
    const color = stock.price >= (stock.previousPrice || stock.price) ? 'green' : 'red';

    const config = `{
            type: 'line',
            data: {
                labels: [${labels}],
                datasets: [{
                    label: '${stock.name}',
                    data: [${data}],
                    borderColor: '${color}',
                    fill: false
                }]
            },
            options: {
                scales: { yAxes: [{ ticks: { beginAtZero: false } }] },
                title: { display: true, text: '${stock.name} (${stock.symbol}) 차트' }
            }
        }`;

    return `https://quickchart.io/chart?c=${encodeURIComponent(config)}`;
  }

  buy(userId: string, symbol: string, amount: number) {
    if (amount <= 0) return { ok: false as const, msg: '수량은 1개 이상이어야 합니다.' };
    const stock = this.stocks[symbol];
    if (!stock) return { ok: false as const, msg: '존재하지 않는 종목입니다.' };

    const user = this.gameData.getUser(userId);
    const totalCost = stock.price * amount;
    if (user.money < totalCost)
      return {
        ok: false as const,
        msg: `돈이 부족합니다. (필요: ${formatMoney(totalCost)}원, 보유: ${formatMoney(user.money)}원)`,
      };

    user.money -= totalCost;
    if (!user.stocks) user.stocks = {};
    if (!user.stocks[symbol]) user.stocks[symbol] = { amount: 0, avgPrice: 0 };
    const us = user.stocks[symbol];
    us.avgPrice = (us.amount * us.avgPrice + amount * stock.price) / (us.amount + amount);
    us.amount += amount;
    this.gameData.saveGameData();
    return {
      ok: true as const,
      stock: stock.name,
      amount,
      price: stock.price,
      totalCost,
      balance: user.money,
    };
  }

  sell(userId: string, symbol: string, amount: number) {
    if (amount <= 0) return { ok: false as const, msg: '수량은 1개 이상이어야 합니다.' };
    const stock = this.stocks[symbol];
    if (!stock) return { ok: false as const, msg: '존재하지 않는 종목입니다.' };

    const user = this.gameData.getUser(userId);
    if (!user.stocks) user.stocks = {};
    const us = user.stocks[symbol];
    if (!us || us.amount < amount) return { ok: false as const, msg: `보유 수량이 부족합니다. (보유: ${us ? us.amount : 0}주)` };

    const totalIncome = stock.price * amount;
    const profit = totalIncome - Math.round(us.avgPrice * amount);
    us.amount -= amount;
    if (us.amount === 0) delete user.stocks[symbol];
    user.money += totalIncome;
    this.gameData.saveGameData();
    return {
      ok: true as const,
      stock: stock.name,
      amount,
      price: stock.price,
      totalIncome,
      profit,
      balance: user.money,
    };
  }

  getPortfolio(userId: string) {
    const user = this.gameData.getUser(userId);
    if (!user.stocks)
      return { entries: [] as { symbol: string; name: string; amount: number; avgPrice: number; currentPrice: number; currentValue: number; profit: number; profitPct: number }[], totalAsset: 0, totalInvested: 0, totalProfit: 0, totalProfitPct: 0 };

    const entries: {
      symbol: string;
      name: string;
      amount: number;
      avgPrice: number;
      currentPrice: number;
      currentValue: number;
      profit: number;
      profitPct: number;
    }[] = [];
    let totalAsset = 0;
    let totalInvested = 0;
    for (const sym in user.stocks) {
      const us = user.stocks[sym];
      const stock = this.stocks[sym];
      if (!stock || us.amount <= 0) continue;
      const cv = stock.price * us.amount;
      const iv = Math.round(us.avgPrice * us.amount);
      totalAsset += cv;
      totalInvested += iv;
      entries.push({
        symbol: sym,
        name: stock.name,
        amount: us.amount,
        avgPrice: us.avgPrice,
        currentPrice: stock.price,
        currentValue: cv,
        profit: cv - iv,
        profitPct: iv > 0 ? ((cv - iv) / iv) * 100 : 0,
      });
    }
    return {
      entries,
      totalAsset,
      totalInvested,
      totalProfit: totalAsset - totalInvested,
      totalProfitPct: totalInvested > 0 ? ((totalAsset - totalInvested) / totalInvested) * 100 : 0,
    };
  }
}

