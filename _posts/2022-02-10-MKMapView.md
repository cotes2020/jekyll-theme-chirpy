---
title: 在 MKMapView 原生地图上显示当前位置的朝向（设备的朝向）
date: 2022-02-10 16:39:12
categories: [iOS]
tags: [MapKit]
---

想在 MKMapView 地图上显示用户位置指向角，苹果官方并没有把 MKModernUserLocationView 暴露给开发者使用，所以只能自己添加朝向的箭头图片。


## 实现

.h文件

``` objc
#import <UIKit/UIKit.h>

@interface LHUserLocationHeadingView : UIView

- (void)startUpdatingHeading;
- (void)stopUpdatingHeading;

@end
```

.m文件

``` objc
#import "LHUserLocationHeadingView.h"
#import <CoreLocation/CoreLocation.h>

@interface LHUserLocationHeadingView()<CLLocationManagerDelegate>

@property (nonatomic, strong) UIImageView *arrowImageView;
@property (nonatomic, strong) CLLocationManager *locationManager;

@end

@implementation AAUserLocationHeadingView

- (instancetype)initWithFrame:(CGRect)frame {
    
    if (self = [super initWithFrame:frame]) {
        [self commonInit];
    }
    
    return self;
}

- (UIImageView *)arrowImageView {
    if (_arrowImageView == nil) {
        _arrowImageView = [[UIImageView alloc] initWithImage:[UIImage imageNamed:@"xz_loction_heading_arrow"]];
    }
    
    return _arrowImageView;
}

- (CLLocationManager *)locationManager {
    if (_locationManager == nil) {
        _locationManager = [[CLLocationManager alloc] init];
        _locationManager.delegate = self;
    }
    
    return _locationManager;
}

- (void)commonInit {
    self.backgroundColor = [UIColor clearColor];
    self.arrowImageView.frame = self.frame;
    [self addSubview:self.arrowImageView];
    [self startUpdatingHeading];
}

- (void)startUpdatingHeading {
    [self.locationManager startUpdatingHeading];
}

- (void)stopUpdatingHeading {
    [self.locationManager stopUpdatingHeading];
}

#pragma mark -- CLLocationManagerDelegate
- (void)locationManager:(CLLocationManager *)manager didUpdateHeading:(CLHeading *)newHeading {
    if (newHeading.headingAccuracy < 0) {
        return;
    }
    
    CLLocationDirection heading = newHeading.trueHeading > 0 ? newHeading.trueHeading : newHeading.magneticHeading;
    CGFloat rotation =  heading / 180 * M_PI;
    self.arrowImageView.transform = CGAffineTransformMakeRotation(rotation);
}

```

<br>

## 使用

在地图初始化时会掉用这个代理方法，可以判断是否是私有对象 `MKModernUserLocationView`，来添加自定义的控件在上面

``` objc
- (void)mapView:(MKMapView *)mapView didAddAnnotationViews (NSArray<MKAnnotationView *> *)views {
    
    if ([views.lastObject isKindOfClass:NSClassFromString(@"MKModernUserLocationView")]) {
        
        AAUserLocationHeadingView *headingView = [[AAUserLocationHeadingView alloc] initWithFrame:CGRectMake(0, 0, 36, 36)];
        headingView.center = CGPointMake(views.lastObject.width/2, views.lastObject.height / 2);
        headingView.tag = 312;
        if (![views.lastObject viewWithTag:312]) {
            [views.lastObject addSubview:headingView];
            _userHeadingView = headingView;
        }
    }  
}
```

<br>

``` objc
// 在Map不在屏幕显示时记得停止获取方向 进入时再开始
[_userHeadingView startUpdatingHeading];
[_userHeadingView stopUpdatingHeading];
```

<br>

``` objc
// 在跟随模式下记得隐藏，其他模式下显示
_userHeadingView.hidden = YES;
``` 
