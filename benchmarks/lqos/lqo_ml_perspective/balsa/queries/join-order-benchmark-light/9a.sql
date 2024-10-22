SELECT COUNT(*) 
FROM cast_info AS ci,
     title AS t,
     movie_companies AS mc 
WHERE t.id = ci.movie_id 
  AND t.id = mc.movie_id 
  AND t.production_year > 2005 
  AND t.production_year < 2015 
  AND ci.role_id = 2;
