-- DEX Weekly Volume (top DEXes)
-- Dune query: tracks weekly DEX trading volume across major protocols
SELECT
    date_trunc('week', block_time) AS week,
    project,
    SUM(amount_usd) AS volume_usd,
    COUNT(*) AS trade_count
FROM dex.trades
WHERE block_time >= NOW() - INTERVAL '90' DAY
GROUP BY 1, 2
ORDER BY 1 DESC, 3 DESC
LIMIT 500
