---
title: Baby Android 1
date: 2025-06-02 15:33:51 -0700
categories: [CTF]
tags: [Rev,jadx,android,apk]
media_subpath: /images/babyandroid1
image: 
  path: baby.png
---
We reversed an APK that instantly wipes the flag on launch. By analyzing the layout XML in Android Studio, we pieced the flag back together from hidden TextViews. This challenge is from  BYUCTF, which is hosted by BYU Cyberia, the official CTF team of Brigham Young University.

# First Look: What Are We Dealing With
![intro](intro.webp)


You're handed an `.apk` file called `baby-android-1.apk`. You should never just start dragging stuff into tools, always peek at the file type first.

```bash
file baby-android-1.apk
```

You'll get something like:

```bash
baby-android-1.apk: Zip archive data, at least v0.0 to extract, compression method=store
```

That means it’s an **Android app package.** Think of it like a `.zip` that contains everything an Android app needs (code, layouts, assets, etc).
## 1: Open It Up

we can use Apktool for decompiling the file with `apktool d **baby-android-1.apk**` . apktool is for reverse engineering third-party, closed, binary, Android apps. Running this will give us:

```bash
I: Using Apktool 2.7.0-dirty on baby-android-1.apk
I: Loading resource table...
I: Decoding AndroidManifest.xml with resources...
I: Loading resource table from file: /home/pollo/.local/share/apktool/framework/1.apk
I: Regular manifest package...
I: Decoding file-resources...
I: Decoding values */* XMLs...
I: Baksmaling classes.dex...
I: Baksmaling classes3.dex...
I: Baksmaling classes2.dex...
I: Copying assets and libs...
I: Copying unknown files...
I: Copying original files...
I: Copying META-INF/services directory

AndroidManifest.xml  apktool.yml  kotlin  META-INF  original  res  smali  smali_classes2  smali_classes3  unknown

                                      
```

When we use apktool on the APK file, it unpacks everything inside: the resources like images and layout XMLs, the manifest file that describes how the app runs, and most importantly, the code, specifically .dex files, which are like compiled bytecode for Android. When Android apps are compiled from Java/Kotlin, they are translated into bytecode.  

Apktool tries to make these .dex files readable by converting them into smali, which is sort of like Android’s version of assembly. But smali is still pretty low-level and hard to follow(for now), so instead, we use a tool called JADX to convert the .dex files directly into Java-like source code, which is way easier to understand. Once we open the .apk in jadx-gui, we get a clean, browsable interface showing us the app’s logic in near-original Java, making it way easier to reverse engineer what’s happening.

open jadx-gui and open the apk
## 2: Understand the App’s Flow

![tree](tree.webp)

when we load into jadx you'll see the jadx tree breakdown. there are many things  but the one that interests me the most is `byuctf.downwiththefrench` because that is the package that contains Main activity, which is the main logic for the app. 

In JADX, go to the folder:

```
byuctf.downwiththefrench → MainActivity
```

This is the main screen of the app. Here's what you’ll see:

![UI](ui.webp)

```java
package byuctf.downwiththefrench;

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

/* loaded from: classes3.dex */
public class MainActivity extends AppCompatActivity {
    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Utilities util = new Utilities(this);
        util.cleanUp();
        TextView homeText = (TextView) findViewById(R.id.homeText);
        homeText.setText("Too slow!!");
    }
}
```

So:

- It  sets a layout called `activity_main.xml`
- Then it runs `cleanUp()` which is a method in `Utilities.java`

Let’s open `Utilities.java`. This is what it does:

```java
package byuctf.downwiththefrench;

import android.app.Activity;
import android.widget.TextView;

/* loaded from: classes3.dex */
public class Utilities {
    private Activity activity;

    public Utilities(Activity activity) {
        this.activity = activity;
    }

    public void cleanUp() {
        TextView flag = (TextView) this.activity.findViewById(R.id.flagPart1);
        flag.setText("");
        TextView flag2 = (TextView) this.activity.findViewById(R.id.flagPart2);
        flag2.setText("");
        TextView flag3 = (TextView) this.activity.findViewById(R.id.flagPart3);
        flag3.setText("");
        TextView flag4 = (TextView) this.activity.findViewById(R.id.flagPart4);
        flag4.setText("");
        TextView flag5 = (TextView) this.activity.findViewById(R.id.flagPart5);
        flag5.setText("");
        TextView flag6 = (TextView) this.activity.findViewById(R.id.flagPart6);
        flag6.setText("");
        TextView flag7 = (TextView) this.activity.findViewById(R.id.flagPart7);
        flag7.setText("");
        TextView flag8 = (TextView) this.activity.findViewById(R.id.flagPart8);
        flag8.setText("");
        TextView flag9 = (TextView) this.activity.findViewById(R.id.flagPart9);
        flag9.setText("");
        TextView flag10 = (TextView) this.activity.findViewById(R.id.flagPart10);
        flag10.setText("");
        TextView flag11 = (TextView) this.activity.findViewById(R.id.flagPart11);
        flag11.setText("");
        TextView flag12 = (TextView) this.activity.findViewById(R.id.flagPart12);
        flag12.setText("");
        TextView flag13 = (TextView) this.activity.findViewById(R.id.flagPart13);
        flag13.setText("");
        TextView flag14 = (TextView) this.activity.findViewById(R.id.flagPart14);
        flag14.setText("");
        TextView flag15 = (TextView) this.activity.findViewById(R.id.flagPart15);
        flag15.setText("");
        TextView flag16 = (TextView) this.activity.findViewById(R.id.flagPart16);
        flag16.setText("");
        TextView flag17 = (TextView) this.activity.findViewById(R.id.flagPart17);
        flag17.setText("");
        TextView flag18 = (TextView) this.activity.findViewById(R.id.flagPart18);
        flag18.setText("");
        TextView flag19 = (TextView) this.activity.findViewById(R.id.flagPart19);
        flag19.setText("");
        TextView flag20 = (TextView) this.activity.findViewById(R.id.flagPart20);
        flag20.setText("");
        TextView flag21 = (TextView) this.activity.findViewById(R.id.flagPart21);
        flag21.setText("");
        TextView flag22 = (TextView) this.activity.findViewById(R.id.flagPart22);
        flag22.setText("");
        TextView flag23 = (TextView) this.activity.findViewById(R.id.flagPart23);
        flag23.setText("");
        TextView flag24 = (TextView) this.activity.findViewById(R.id.flagPart24);
        flag24.setText("");
        TextView flag25 = (TextView) this.activity.findViewById(R.id.flagPart25);
        flag25.setText("");
        TextView flag26 = (TextView) this.activity.findViewById(R.id.flagPart26);
        flag26.setText("");
        TextView flag27 = (TextView) this.activity.findViewById(R.id.flagPart27);
        flag27.setText("");
        TextView flag28 = (TextView) this.activity.findViewById(R.id.flagPart28);
        flag28.setText("");
    }
}
```

It’s wiping out **28 TextViews** that probably contain the flag.  At this point in a bit confused since this doesn't seem like a functional feature for an app.  Lets open up an android on Genymotion and upload the apk file to it. Genymotion is just an android emulator that I like using. 

We can see that the uploaded apk is named “Down With the french”. When i run the app all  i see is “Too slow!!”. Maybe is says this because of the 28 flags that it wiped out when its launched. You never see it because the app clears it instantly when you launch it. Lets dig deeper into the layout. 

<div style="display: flex; justify-content: center; gap: 1rem; margin: 2rem 0;">
  <img src="left.webp" alt="Left" style="width: 45%; border-radius: 8px;" />
  <img src="right.webp" alt="Right" style="width: 45%; border-radius: 8px;" />
</div>


---

## 3: We Need the Layout

Since the text is set in the layout, not code, we want to read `activity_main.xml`. as we recall with this line of code in main `setContentView(R.layout.activity_main);` 

To get access to layout files we can go to the tree on the left and go to `Resources —> res —> layout —> activity_main.xml`

> When you're reversing an Android app and you want to know what shows up on screen when the app launches, the layout file is almost always:  
> —> res —> layout —> activity_main.xml  
{: .prompt-tip }

And now we see a **bunch of TextViews**, like this:

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android" xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">
    <TextView
        android:id="@+id/homeText"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="b"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.066"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.022"/>
    <TextView
        android:id="@+id/flagPart1"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="420dp"
        android:text="}"
        android:layout_marginEnd="216dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"/>
    <TextView
        android:id="@+id/flagPart2"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="616dp"
        android:text="t"
        android:layout_marginEnd="340dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"/>
    <TextView
        android:id="@+id/flagPart3"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="556dp"
        android:text="a"
        android:layout_marginEnd="332dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"/>
    <TextView
        android:id="@+id/flagPart4"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="676dp"
        android:text="y"
        android:layout_marginEnd="368dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"/>
    <TextView
        android:id="@+id/flagPart5"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="500dp"
        android:text="c"
        android:layout_marginEnd="252dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"/>
    <TextView
        android:id="@+id/flagPart6"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginBottom="636dp"
        android:text="c"
        android:layout_marginEnd="348dp"
        app:layout_constraintBottom_toBottomOf="parent"

```

This is where the flag is hiding!

---

### It’s Not in Order?

You might assume `flagPart1`, `flagPart2`, etc. are in order. **They’re not**.

They’re placed on screen by **margin values,**  meaning how far they appear from the bottom (vertical position). So the **higher the number in `layout_marginBottom`**, the **higher it appears on screen**.

So to reconstruct the flag, we need to sort the TextViews **top-to-bottom** based on their vertical positions. In this case It’ll be way easier if we can physically see how its ordered on a screen. 

---

## 4: Use Android Studio to Visualize It

This is the easiest way to render the layout XML and  "see" how the UI looks without running the actual app. Android studio design view will just parse the XML and show what it would look like. 

### Here’s what you do:

1. **Open Android Studio**
2. Create a new project → choose **Empty views activity**
3. once all loaded in, In the side panel, go to:

![studio](studio.webp)

```
app → res → layout → activity_main.xml
```

1. **Replace all of that file’s contents** with the `activity_main.xml` from the `activity_main_.xl from your apk`. 
2. Click the **“Design” tab** on the top right.
    
    ![flag](xml.webp)
    

the full UI is rendered, and all the flag parts are arranged.

Now you can **read the flag top-to-bottom how it would appear on screen.** 

---

# Final Flag

From the layout preview, you’ll see:

```
byuctf{android_piece_0f_c4ke}
```
---