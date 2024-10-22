SELECT COUNT(*) 
FROM title AS t,
     movie_keyword AS mk,
     movie_companies AS mc,
     movie_info AS mi 
WHERE t.id = mk.movie_id 
  AND t.id = mc.movie_id 
  AND t.id = mi.movie_id 
  AND mk.keyword_id = 398 
  AND mc.company_type_id = 2 
  AND t.production_year > 2000 
  AND t.production_year < 2010;
