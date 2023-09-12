---
title: NSAttributedString 属性文本（图文混排）
date: 2018-04-27 15:14:17
categories: iOS
tags: 零散
---

## NSMutableAttributedString 的使用
* 唯一蛋疼的是无法实现监听文本中链接、话题等的点击事件
* 如果要实现监听链接、话题等的点击事件，可以使用 [YYText](https://github.com/ibireme/YYText) 框架，或直接使用 [YYKit](https://github.com/ibireme/YYKit)

<br>

``` objc
@interface ViewController ()

@property (nonatomic, strong) UILabel *textLabel;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.textLabel = [[UILabel alloc] initWithFrame:CGRectMake(50, 100, 200, 80)];
    self.textLabel.backgroundColor = [UIColor grayColor];
    self.textLabel.lineBreakMode = NSLineBreakByWordWrapping;
    self.textLabel.font = [UIFont systemFontOfSize:15];
    self.textLabel.numberOfLines = 0;
    [self.view addSubview:self.textLabel];
    
    NSString *text = @"就😢发就发就发就发aaaaaaa就就发就就发就发就发就发了是就发就发就发就发aaaaaaa就就发就就发就发就发就发了是就发就发就发就发aaaaaaa就就发就就发就发就发就发了是\n看到就😢😢😢发了就发就发就发就发就发aaaaaaa就就发就就发就发就发就发了是就发就发就发aaaaaaa就就发就就发就发就发就发了是上课点击发😢😢😢😢😢😢";
    
    NSMutableParagraphStyle *paragraphStyle = [[NSMutableParagraphStyle alloc] init];
    paragraphStyle.lineSpacing = 2.5;
    paragraphStyle.paragraphSpacing = 7.5;
    
    // 段落首行缩进
    paragraphStyle.firstLineHeadIndent = 30;//26.5;
    // paragraphStyle.lineBreakMode = NSLineBreakByCharWrapping;
    // paragraphStyle.maximumLineHeight = 20;            // 文本最大行距
    // paragraphStyle.minimumLineHeight = 20;
    // paragraphStyle.alignment = NSTextAlignmentJustified;
    // paragraphStyle.paragraphSpacingBefore = 20;
    
    // 这个字典里要一定包含字体大小，其他的看具体需求
    UIFont *font = [UIFont systemFontOfSize:15];
    NSDictionary *dic = @{
                          NSFontAttributeName : font,
                          NSParagraphStyleAttributeName : paragraphStyle
                          };
    
    NSDictionary *dic2 = @{
                          NSForegroundColorAttributeName : [UIColor redColor],
                          NSFontAttributeName : [UIFont systemFontOfSize:50]
                          };
    
    
    NSMutableAttributedString *attString = [[NSMutableAttributedString alloc] initWithString:text attributes:dic];
    // [attString addAttributes:dic2 range:NSMakeRange(2, 3)];
    
    NSTextAttachment *attachment = [[NSTextAttachment alloc] init];
    attachment.image = [UIImage imageNamed:@"test"];
    
    // 设置 y 值是为了让图片和文字对齐
    attachment.bounds = CGRectMake(0, -3, font.lineHeight, font.lineHeight);
    
    NSAttributedString *textTttachment = [NSAttributedString attributedStringWithAttachment:attachment];
    [attString insertAttributedString:textTttachment atIndex:1];
    [attString appendAttributedString:textTttachment];
    
    // options 一般都写：NSStringDrawingUsesFontLeading | NSStringDrawingUsesLineFragmentOrigin
    // 官方文档说要用 ceil 方法搞一下。
    CGSize size = [attString boundingRectWithSize:CGSizeMake(200, CGFLOAT_MAX) options:NSStringDrawingUsesFontLeading | NSStringDrawingUsesLineFragmentOrigin context:nil].size;
    double height = ceil(size.height);
    
    if (size.height > 1) {
        
    }
    
    self.textLabel.frame = CGRectMake(50, 100, 200, height);
    self.textLabel.attributedText = attString;
    
    // 给 UILabel 设置 attributedText 后，超出的文字不显示省略号，可能是设置以后 UILabel
    // 原本的设置会被覆盖(包括字体,字体颜色等属性)，如果要显示省略号，得重新设置一下
    self.textLabel.lineBreakMode = NSLineBreakByTruncatingTail;
}

```
