//Встраивание: public BScreen ScreenR = new BScreen(); - инициализация
//ScreenR.ScreenRec(); - разрешение открыть поток
//ScreenR.recFrameFlag = true; - разрешение записи
//StartCoroutine(ScreenR.RecordFrame()); - запись фрейма при истинном флаге выше
//ScreenR.ScreenRec(); - закрытие потока обязательно

using UnityEditor.Media;
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BScreen
{
    public bool recFlag;
    public bool recFrameFlag;
    private int widthF;
    private int heightF;
    private string filePath;
    private string fileName;
    private MediaEncoder encoder;
    private int frameCnt;

    public BScreen() {
        this.recFlag = false;
        this.recFrameFlag = false;
        this.widthF = 670;  //320 //774
        this.heightF = 290; //200 //347
        //this.filePath = @"G:\Other\3Dfuck\1_PC\Screenshots\";
        this.filePath = @"G:\Other\3Dfuck\TestingServer\Screenshots\";
        //this.filePath = "~/NewProjects/gdfjvdkfuv/movies";
        this.fileName = "my_movie.mp4";
        this.frameCnt = 0;
    }

    public IEnumerator RecordFrame()
    {
        yield return new WaitForEndOfFrame();
        if(this.recFrameFlag) //Именно условие не протестировано, убрать если не заработает. Код в скобках работает.
        {
            Texture2D texture = new Texture2D(this.widthF, this.heightF, TextureFormat.RGBA32, false);
            Texture2D tex = new Texture2D(this.widthF, this.heightF, TextureFormat.RGBA32, false);
            texture = ScreenCapture.CaptureScreenshotAsTexture();
            // do something with texture

            for (int y = 0; y < this.heightF; y++)
            {
                for (int x = 0; x < this.widthF; x++)
                {
                    tex.SetPixel(x,y,texture.GetPixel(x,y));
                }
            }
            //tex.Apply();
            this.frameCnt++;
            Debug.Log("Frame!" + this.frameCnt);
            this.encoder.AddFrame(tex);
            // cleanup
            UnityEngine.Object.Destroy(tex);
            UnityEngine.Object.Destroy(texture);
        }
    }

    public void ScreenRec()
    {
        if(!this.recFlag)
        {
            var encodedFilePath = Path.Combine(Path.GetDirectoryName(this.filePath),this.fileName);
            encodedFilePath = Path.GetFullPath(encodedFilePath);
            //var encodedFilePath = AppDomain.CurrentDomain.BaseDirectory;
            //encodedFilePath = Path.Combine(encodedFilePath,this.fileName); //Для Mac и unix  систем раскоментить.
            //Debug.Log(encodedFilePath);

                var videoAttr = new VideoTrackAttributes
                {
                    frameRate = new MediaRational(30),
                    width = (uint)widthF,
                    height = (uint)heightF,
                    includeAlpha = false
                };
                this.encoder = new MediaEncoder(encodedFilePath, videoAttr);
        }
        else
        {
            this.encoder.Dispose();
        }
        this.recFlag = !this.recFlag;
    }

   /* public void LateUpdate()
    {
        if(this.recFlag)
        {
            StartCoroutine(RecordFrame());
        }
    }*/
}