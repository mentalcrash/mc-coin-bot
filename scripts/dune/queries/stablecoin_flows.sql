-- Stablecoin Mint/Burn Flows
-- Dune query: tracks daily stablecoin supply changes (mint/burn events)
SELECT
    date_trunc('day', block_time) AS day,
    symbol,
    SUM(CASE WHEN "from" = 0x0000000000000000000000000000000000000000 THEN amount_usd ELSE 0 END) AS minted_usd,
    SUM(CASE WHEN "to" = 0x0000000000000000000000000000000000000000 THEN amount_usd ELSE 0 END) AS burned_usd,
    SUM(CASE WHEN "from" = 0x0000000000000000000000000000000000000000 THEN amount_usd ELSE 0 END)
    - SUM(CASE WHEN "to" = 0x0000000000000000000000000000000000000000 THEN amount_usd ELSE 0 END) AS net_flow_usd
FROM tokens.transfers
WHERE block_time >= NOW() - INTERVAL '30' DAY
    AND symbol IN ('USDT', 'USDC', 'DAI', 'BUSD')
GROUP BY 1, 2
ORDER BY 1 DESC, 2
