# database 题目：

1. 查询Student表中的所有记录的Sname. Ssex和Class列。
2. 查询教师所有的单位即不重复的Depart列。
3. 查询Student表的所有记录。
4. 查询Score表中成绩在60到80之间的所有记录。
5. 查询Score表中成绩为85，86或88的记录。
6. 查询Student表中“95031”班或性别为“女”的同学记录。
`select * from STUDENT where SSEX='女'and CLASS='95031';`

7. 以Class降序查询Student表的所有记录。
`select * from STUDENT ORDER BY class DESC;`

8. 以Cno升序. Degree降序查询Score表的所有记录。
`SELECT * FROM SCORE ORDER BY CNO ASC,DEGREE DESC;`

9. 查询“95031”班的学生人数。
`select class, count(sno) from STUDENT where class='95031';`

1.  查询Score表中的最高分的学生学号和课程号。
`SELECT SNO,CNO FROM SCORE where degree =(SELECT max(DEGREE) FROM SCORE);`
`SELECT SNO,CNO FROM SCORE ORDER BY DEGREE DESC LIMIT 1;`
11. 查询‘3-105’号课程的平均分。
`SELECT avg(degree) FROM SCORE where CNO ='3-105';`
12. 查询Score表中至少有5名学生选修的并以3开头的课程的平均分数。

```
select avg(degree), cno from score
where cno like '3%'group by cno
having count(sno)>= 5;
```
13. 查询最低分大于70，最高分小于90的Sno列。
`SELECT sno FROM SCORE group by SNO HAVING max(degree)<90 and min(degree)>70 ;`
14. 查询所有学生的Sname. Cno和Degree列。
`SELECT sname,cno,DEGREE from STUDENT,SCORE where student.sno=score.sno;`
`SELECT A.SNAME,B.CNO,B.DEGREE FROM STUDENT AS A JOIN SCORE AS B ON A.SNO=B.SNO;`
15. 查询所有学生的Sno. Cname和Degree列。
`SELECT A.CNAME, B.SNO,B.DEGREE FROM COURSE AS A JOIN SCORE AS B ON A.CNO=B.CNO ;`
16. 查询所有学生的Sname. Cname和Degree列。
`SELECT A.SNAME,B.CNAME,C.DEGREE FROM STUDENT A JOIN (COURSE B,SCORE C) ON A.SNO=C.SNO AND B.CNO =C.CNO;`
17. 查询“95033”班所选课程的平均分。
`SELECT AVG(A.DEGREE) FROM SCORE A JOIN STUDENT B ON A.SNO = B.SNO WHERE B.CLASS='95033';`
18. 假设使用如下命令建立了一个grade表：
create table grade(low number(3,0), upp number(3),rank char(1));
insert into grade values(90,100,'A');
insert into grade values(80,89,'B');
insert into grade values(70,79,'C');
insert into grade values(60,69,'D');
insert into grade values(0,59,'E');
commit;

现查询所有同学的Sno. Cno和rank列。
`SELECT A.sno,A.cno,B.rank from SCORE as A JOIN grade AS B;`

19. 查询选修“3-105”课程的成绩高于“109”号同学成绩的所有同学的记录。
`SELECT * from STUDENT WHERE sno in (SELECT SNO from SCORE where cno='3-105' and degree>(SELECT degree from SCORE where sno=109 and cno='3-105'));`
`SELECT A.* FROM SCORE A JOIN SCORE B WHERE A.CNO='3-105' AND A.DEGREE>B.DEGREE AND B.SNO='109' AND B.CNO='3-105';`

20. 查询score中选学一门以上课程的同学中分数为非最高分成绩的记录。
`select * from SCORE where degree<(SELECT max(degree) from SCORE) GROUP BY SNO having count(cno)>1 order by degree;`

21. 查询成绩高于学号为“109”. 课程号为“3-105”的成绩的所有记录。
`select * FROM SCORE WHERE degree>(SELECT degree from SCORE where sno=109 and cno='3-105');`

22. 查询和学号为108的同学同年出生的所有学生的Sno. Sname和Sbirthday列。
`select sno,sname,sbirthday from STUDENT WHERE SBIRTHDAY=(SELECT SBIRTHDAY from STUDENT WHERE sno=108);`

23. 查询“张旭“教师任课的学生成绩。
`SELECT sno,degree from SCORE WHERE cno=(SELECT cno FROM COURSE WHERE TNO=(SELECT tno from TEACHER WHERE tname=='张旭'));`
`SELECT A.SNO,A.DEGREE FROM SCORE A JOIN (TEACHER B,COURSE C) ON (A.CNO=C.CNO AND B.TNO=C.TNO) WHERE B.TNAME='张旭';`

24. 查询选修某课程的同学人数多于5人的教师姓名。
`SELECT A.tname FROM TEACHER A JOIN (COURSE B, SCORE C) ON (A.tno=B.tno and B.cno=c.cno) group by c.cno having count(c.sno)>5;`

25. 查询95033班和95031班全体学生的记录。
`SELECT cno FROM SCORE WHERE degree>85 group by CNO;`
26. 查询存在有85分以上成绩的课程Cno.
`SELECT cno FROM SCORE WHERE degree>85 group by CNO;`
`SELECT CNO FROM SCORE GROUP BY CNO HAVING MAX(DEGREE)>85;`

27. 查询出“计算机系“教师所教课程的成绩表。
`SELECT A.* from SCORE A JOIN (COURSE b, TEACHER C) ON (A.cno=B.cno and B.TNO=C.tno) WHERE C.depart='计算机系';`
`SELECT * from score where cno in (select a.cno from course a join teacher b on a.tno=b.tno and b.depart='计算机系');`
//此时2略好于1，在多连接的境况下性能会迅速下降

28. 查询“计算机系”与“电子工程系“不同职称的教师的Tname和Prof。

29. 查询选修编号为“3-105“课程且成绩至少高于选修编号为“3-245”的同学的Cno. Sno和Degree,并按Degree从高到低次序排序。
`ELECT cno,sno,degree FROM SCORE where CNO='3-105' AND degree>(SELECT degree FROM SCORE where cno='3-245') ORDER BY degree desc;`
`SELECT cno,sno,degree FROM SCORE WHERE DEGREE>ALL(SELECT DEGREE FROM SCORE WHERE CNO='3-245') ORDER BY DEGREE DESC;`

30. 查询选修编号为“3-105”课程且成绩高于选修编号为“3-245”课程的同学的Cno. Sno和Degree.
`SELECT * FROM SCORE WHERE DEGREE>ALL(SELECT DEGREE FROM SCORE WHERE CNO='3-245') ORDER BY DEGREE DESC;`

31. 查询所有教师和同学的name. sex和birthday.

```
SELECT SNAME AS NAME, SSEX AS SEX, SBIRTHDAY AS BIRTHDAY FROM STUDENT
UNION
SELECT TNAME AS NAME, TSEX AS SEX, TBIRTHDAY AS BIRTHDAY FROM TEACHER;
```

32. 查询所有“女”教师和“女”同学的name. sex和birthday.

```
SELECT sname AS name, ssex AS sex, sbirthday as birthday FROM STUDENT WHERE ssex='女'
UNION
SELECT tname AS name, tsex AS sex, tbirthday as birthday FROM TEACHER WHERE tsex='女';
```

33. 查询成绩比该课程平均成绩低的同学的成绩表。
`SELECT A.* FROM SCORE A WHERE DEGREE<(SELECT AVG(DEGREE) FROM SCORE B WHERE A.CNO=B.CNO);`

34. 查询所有任课教师的Tname和Depart.
35  查询所有未讲课的教师的Tname和Depart.
36. 查询至少有2名男生的班号。
37. 查询Student表中不姓“王”的同学记录。
38. 查询Student表中每个学生的姓名和年龄。
39. 查询Student表中最大和最小的Sbirthday日期值。
40. 以班号和年龄从大到小的顺序查询Student表中的全部记录。
41. 查询“男”教师及其所上的课程。
42. 查询最高分同学的Sno. Cno和Degree列。
43. 查询和“李军”同性别的所有同学的Sname.
44. 查询和“李军”同性别并同班的同学Sname.
45. 查询所有选修“计算机导论”课程的“男”同学的成绩表
