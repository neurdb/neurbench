-- TPC TPC-H Parameter Substitution (Version 2.17.3 build 0)
-- using default substitutions

select
	s_suppkey,
	s_name,
	s_address,
	s_phone,
	total_revenue
from
	supplier,
	revenue_train
where
	s_suppkey = supplier_no
	and total_revenue = (
		select
			max(total_revenue)
		from
			revenue_train
	)
order by
	s_suppkey