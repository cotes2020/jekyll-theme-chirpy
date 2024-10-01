---
title: Further Adventures in SVGs
description: Creating a trend SVG visual in Power BI
author: duddy
date: 2024-10-01 12:00:00 +0000
categories: [SVG, DAX]
tags: [svg, dax]
pin: false
image:
  path: /assets/img//0009-SVGTrend/SVGTrendSmall.png
  alt: Example of Trend SVG
---

In a previous post I created a [SVG Barbell visual](https://evaluationcontext.github.io/posts/SVG-RAG-Dumbell/). While this was a good start I wanted to expand upon this concept by adding an extra dimension of time to see trends. The results are below.

```dax
Trend SVG =
VAR _SvgWidth =                 100
VAR _SvgHeight =                20

// values
VAR _ActualValue =              [Max Value]
VAR _ActualColour=              [Colour Hex]
VAR _AcutalValueFormatted =     IF( MAX( Metrics[format] ) = "Percent", FORMAT( _ActualValue, "0.0%"), FORMAT( _ActualValue, "0.0") )
VAR _RedValue =                 MAX( Metrics[red#] )
VAR _GreenValue =               MAX( Metrics[green#] )

// y axis
VAR _Actuals =
    ADDCOLUMNS(
        CALCULATETABLE( VALUES( Periods[Reporting Period] ), ALLSELECTED( Periods ) )
        ,"@Actual", CALCULATE( [Max Value] )
    )

VAR _MinActual =                MINX( _Actuals, [@Actual] )
VAR _MaxActual =                MAXX( _Actuals, [@Actual] )
VAR _SmallValue =               MIN( _RedValue, _GreenValue)
VAR _LargeValue =               MAX( _RedValue, _GreenValue) 
VAR _SmallestValue =            MIN( _SmallValue, _MinActual )
VAR _LargestValue =             MAX( _LargeValue, _MaxActual )

VAR _YOffset =                  5
VAR _YInputStart =              _SmallestValue                     // The lowest number of the range input
VAR _YInputEnd =                _LargestValue                      // The largest number of the range input
VAR _YOutputStart =             _SvgHeight -  _YOffset             // The lowest number of the range output
VAR _YOutputEnd =               0 + _YOffset                        // The largest number of the range output

// x axis
VAR _DateMin =                  CALCULATE( MINX( SUMMARIZE( 'Fact', Periods[End Date] ), Periods[End Date] ), ALLSELECTED( Periods ), ALLSELECTED( Metrics ) )
VAR _DateMax =                  CALCULATE( MAXX( SUMMARIZE( 'Fact', Periods[End Date] ), Periods[End Date] ), ALLSELECTED( Periods ), ALLSELECTED( Metrics ) )
VAR _MaxDateWithAcutal =        CALCULATE( MAXX( SUMMARIZE( 'Fact', Periods[End Date] ), Periods[End Date] ) )

VAR _XOffset =                   5
VAR _XInputStart =               _DateMin                           // The lowest number of the range input
VAR _XInputEnd =                 _DateMax                           // The largest number of the range input
VAR _XOutputStart =              0 + _XOffset                       // The lowest number of the range output
VAR _XOutputEnd =                _SvgWidth - _XOffset - 20          // The largest number of the range output

// Colours
VAR _Opacity =                  "73" // 45%
VAR _RedHex =                   "#A9000A"
VAR _AmberHex =                 "#E49F16"
VAR _GreenHex =                 "#00847E"
VAR _GreyHex =                  "#A3A3A3"
VAR _BlackHex =                 "#000000"
VAR _SmallHex =                 IF( _GreenValue = _SmallValue, _GreenHex, _RedHex )
VAR _LargeHex =                 IF( _GreenValue = _LargeValue, _GreenHex, _RedHex )
VAR _CallOutHex =               _BlackHex

// Vectors
VAR _TopPosition =             _YOutputStart + ((_YOutputEnd - _YOutputStart) / (_YInputEnd - _YInputStart)) * (_LargeValue  - _YInputStart)
VAR _BottomPosition =          _YOutputStart + ((_YOutputEnd - _YOutputStart) / (_YInputEnd - _YInputStart)) * (_SmallValue  - _YInputStart)
VAR _TopLine =                  "<line x1=""" & _XOutputStart & """" & UNICHAR(10) & "y1=""" & _TopPosition & """ x2=""" & _XOutputEnd & """ y2=""" & _TopPosition & """ stroke=""" & _LargeHex & _Opacity & """/>"
VAR _BottomLine =               "<line x1=""" & _XOutputStart & """" & UNICHAR(10) & "y1=""" & _BottomPosition & """ x2=""" & _XOutputEnd & """ y2=""" & _BottomPosition & """ stroke=""" & _SmallHex & _Opacity & """/>"

// Circles
var _SmallCircleSize =          1.
var _LargeCircleSize =          3
var _Circles =
                                ADDCOLUMNS(
                                    CALCULATETABLE( SUMMARIZE( 'Fact', Periods[End Date] ) ,REMOVEFILTERS( Periods ), DATESBETWEEN( Dates[Date], _DateMin, _MaxDateWithAcutal ) )
                                    ,"@circles"
                                        ,var xVal = Periods[End Date]
                                        var yVal =  CALCULATE( [Max Value], REMOVEFILTERS( Periods ), TREATAS( { xVal }, Periods[End Date] ))
                                        var hex =   CALCULATE( [Colour Hex], REMOVEFILTERS( Periods ), TREATAS( { xVal }, Periods[End Date] ))
                                        var x =     _XOutputStart + ((_XOutputEnd - _XOutputStart) / (_XInputEnd - _XInputStart)) * (xVal - _XInputStart)
                                        var y =     _YOutputStart + ((_YOutputEnd - _YOutputStart) / (_YInputEnd - _YInputStart)) * (yVal  - _YInputStart)
                                        var size =  IF( Periods[End Date] = _MaxDateWithAcutal, _LargeCircleSize, _SmallCircleSize )
                                        return
                                        IF( not ISBLANK( xVal ), "<circle cx=""" & x &""" cy=""" & y & """ r=""" & size & """ fill=""" & IF( Periods[End Date] = _MaxDateWithAcutal, hex, hex & _Opacity ) & """ stroke= '" & IF( Periods[End Date] = _MaxDateWithAcutal, _BlackHex, _BlackHex & _Opacity )  & "'/>", "")
                                        & IF( Periods[End Date] = _MaxDateWithAcutal  , "<text x='" & _SvgWidth - 20 & "' y='"& y + _LargeCircleSize & "' fill='" & _CallOutHex  & "' font-size='8' font-family='Segoe UI, sans-serif' >"& _AcutalValueFormatted &"</text>", "")
                                )               

VAR _Svg =
    "data:image/svg+xml;utf8, <svg width=""" & _SvgWidth & """ height=""" & _SvgHeight &""" xmlns=""http://www.w3.org/2000/svg"">" &
    _TopLine &
    _BottomLine &
    CONCATENATEX( _Circles, [@circles] ) &
    "</svg>"

RETURN
    IF( not ISBLANK( _ActualValue ) &&  not ISBLANK( MAX( Metrics[RAG] )), _Svg )
```

Resulting in the following:

![SVG Dumbbell](/assets/img//0009-SVGTrend/SVGTrendLarge.png)

I have seen some interesting examples of SVGs that utilize css and classes. SVG's still have plenty of depth that I have yet to dive into.