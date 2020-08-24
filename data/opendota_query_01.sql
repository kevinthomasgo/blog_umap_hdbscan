SELECT
        
matches.match_id,
matches.start_time,
((player_matches.player_slot < 128) = matches.radiant_win) win,
player_matches.hero_id,
player_matches.account_id,
leagues.name leaguename,

kills,
deaths,
assists,
gold,
last_hits,
denies,
gold_per_min,
xp_per_min,
gold_spent,
hero_damage,
tower_damage,
hero_healing,
level,
stuns,
obs_placed,
sen_placed,
creeps_stacked,
camps_stacked,
rune_pickups,
lane,
lane_role,
is_roaming,
firstblood_claimed,
teamfight_participation,
towers_killed,
roshans_killed


FROM matches
JOIN match_patch using(match_id)
JOIN leagues using(leagueid)
JOIN player_matches using(match_id)
JOIN heroes on heroes.id = player_matches.hero_id
LEFT JOIN notable_players ON notable_players.account_id = player_matches.account_id
LEFT JOIN teams using(team_id)
WHERE
(matches.leagueid = 10826 OR matches.leagueid = 10810 OR matches.leagueid = 10681 OR matches.leagueid = 10153 OR matches.leagueid = 10482 OR matches.leagueid = 11280 OR matches.leagueid = 10749)
-- AND matches.match_id = 4967602648
-- AND notable_players.team_id = 2586976
ORDER BY matches.match_id, win 
-- LIMIT 200