-- Whale Transfers (BTC/ETH large transfers)
-- Dune query: tracks large token transfers (> $1M USD equivalent)
SELECT
    block_time,
    "from" AS sender,
    "to" AS receiver,
    symbol,
    amount,
    amount_usd
FROM tokens.transfers
WHERE block_time >= NOW() - INTERVAL '7' DAY
    AND amount_usd > 1000000
    AND symbol IN ('WBTC', 'WETH', 'USDT', 'USDC')
ORDER BY amount_usd DESC
LIMIT 200
