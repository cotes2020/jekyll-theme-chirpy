interface EXIFStatic {
    getData(url: string, callback: any): any;
    getTag(img: any, tag: any): any;
    getAllTags(img: any): any;
    pretty(img: any): string;
    readFromBinaryFile(file: any): any;
}

declare var EXIF : EXIFStatic;
export = EXIF;
