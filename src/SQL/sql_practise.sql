#-------------------------------------------------------------
SELECT Id,
       Company,
       Salary
FROM
  (SELECT *, row_number() over(PARTITION BY Company
                            ORDER BY Salary ASC,Id ASC) rnum_asc,
                       row_number() over(PARTITION BY Company
                                         ORDER BY Salary DESC,Id DESC) rnum_desc
   FROM Numbers) t
WHERE cast(rnum_asc AS signed)-cast(rnum_desc AS signed) in (1,0,-1)
ORDER BY Company, Salary;


SELECT round(avg(avg_dp), 2) average_daily_percent
FROM
  (SELECT action_date,
          (count(DISTINCT r.post_id)/ count(DISTINCT a.post_id)) *100 AS avg_dp
   FROM Actions a
   LEFT OUTER JOIN Removals r ON a.post_id = r.post_id
   WHERE extra = 'spam'
   GROUP BY a.action_date) t

#---------------------------------------------------------------------------------------------
