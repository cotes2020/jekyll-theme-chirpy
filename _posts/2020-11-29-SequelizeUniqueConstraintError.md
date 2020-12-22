---
title: SequelizeUniqueConstraintError
author: juyoung
date: 2020-11-29 19:05:00 +0800
categories: [project, error]
tags: [error]
---


![followError.jpg](/assets/img/followError.jpg)

![validationError.jpg](/assets/img/validationError.jpg)  


```console
UniqueConstraintError [SequelizeUniqueConstraintError]: Validation error
    at Query.formatError (C:\Users\HOME\Documents\ymillonga\back\node_modules\sequelize\lib\dialects\mysql\query.js:218:16)   
    at Query.run (C:\Users\HOME\Documents\ymillonga\back\node_modules\sequelize\lib\dialects\mysql\query.js:54:18)
    at processTicksAndRejections (internal/process/task_queues.js:93:5)
    at async C:\Users\HOME\Documents\ymillonga\back\node_modules\sequelize\lib\sequelize.js:619:16
    at async MySQLQueryInterface.bulkInsert (C:\Users\HOME\Documents\ymillonga\back\node_modules\sequelize\lib\dialects\abstract\query-interface.js:818:21)
    at async recursiveBulkCreate (C:\Users\HOME\Documents\ymillonga\back\node_modules\sequelize\lib\model.js:2698:25)
    at async Function.bulkCreate (C:\Users\HOME\Documents\ymillonga\back\node_modules\sequelize\lib\model.js:2824:12)
    at async Promise.all (index 0)
    at async BelongsToMany.add (C:\Users\HOME\Documents\ymillonga\back\node_modules\sequelize\lib\associations\belongs-to-many.js:740:30)
    at async C:\Users\HOME\Documents\ymillonga\back\routes\user.js:168:9 {
  errors: [
    ValidationErrorItem {
      message: 'follow.PRIMARY must be unique',
      type: 'unique violation',
      path: 'follow.PRIMARY',
      value: '2-1',
      origin: 'DB',
      instance: null,
      validatorKey: 'not_unique',
      validatorName: null,
      validatorArgs: []
    }
  ],
  fields: { 'follow.PRIMARY': '2-1' },
  parent: Error: Duplicate entry '2-1' for key 'follow.PRIMARY'      at Packet.asError 
```


해결방법:
제로초님이 네트워크에 요청이 두번 가고 있다고 해서 혹시나하고 사가를 확인해보니
userSaga에서 fork를 두번하고 있었다...

export default function* userSaga() {
    yield all([
        fork(watchLoadFollowers),
        fork(watchLoadFollowings),
        fork(watchLoadUser),
        fork(watchSignup),
        fork(watchChangeNickname),
        fork(watchFollow),
		 fork(watchFollow),
        fork(watchUnfollow),
        fork(watchRemovefollower),
        fork(watchLogin),
        fork(watchLogout),
    ]);

}
