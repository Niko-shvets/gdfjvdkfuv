/*
add this file to unity project 

1. add character 
2. create animator for this character
3. append this script to the character
*/




using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Data;

using System;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;

// [ExecuteInEditMode]
public class SimpleMove : MonoBehaviour
{      

    // public GameObject button;
    // public Button btn;
    // public Button button;

    Animator animator;

    private GameObject BodyTarget = null;
    private GameObject NeckTarget = null;
    private GameObject BodyRotationTarget = null;
    private GameObject RKneeTarget = null;
    private GameObject RFootTarget = null;
    private GameObject LKneeTarget = null;
    private GameObject LFootTarget = null;
    private GameObject RElbowTarget = null;
    private GameObject RHandTarget = null;
    private GameObject LElbowTarget = null;
    private GameObject LHandTarget = null;
    private GameObject RShoulderTarget = null;
    private GameObject LShoulderTarget = null;
    private GameObject RHipTarget = null;
    private GameObject LHipTarget = null;
        
    // angles fot rotation of body parts;
    private float RShoulderAngle; 
    private float RElbowAngle; 
    private float LShoulderAngle;
    private float LElbowAngle;
    private float LHipAngle;
    private float LKneeAngle;
    private float RHipAngle;
    private float RKneeAngle;
    private float HeadAngle_y;
    private float BodyAngle_y;

    // Init vector position of each body part;
    Vector3 AssPos;
    Vector3 NeckPos;
    Vector3 ChestPos;
    Vector3 RHipPos;
    Vector3 RKneePos;
    Vector3 RAnklePos;
    Vector3 LHipPos;
    Vector3 LKneePos;
    Vector3 LAnklePos;
    Vector3 RShoulderPos;
    Vector3 RForeArmPos;
    Vector3 RWristPos;
    Vector3 LShoulderPos;
    Vector3 LForeArmPos;
    Vector3 LWristPos;
    Vector3 LWristDirection;

    // Thread
    Thread receiveThread;
    TcpClient client;
    TcpListener listener;
    // int port = 5066; 
    int port = 8052;


    // Start is called before the first frame update
    void Start()
    {
        // button = GameObject.Find("Button").GetComponent<Button>();

        // LHandTarget1 = GameObject.Find("LHandTarget(1)");
        // Debug.Log(LHandTarget1);

        animator = GetComponent<Animator>();

        // create targets for IK: 
        // body targets
        BodyTarget = new GameObject("BodyTarget");
        BodyTarget.transform.rotation = Quaternion.identity;

        NeckTarget = new GameObject("NeckTarget");
        NeckTarget.transform.SetParent(BodyTarget.transform, false);
        NeckTarget.transform.rotation = Quaternion.identity;

        BodyRotationTarget = new GameObject("BodyRotationTarget");
        BodyRotationTarget.transform.SetParent(BodyTarget.transform, false);
        BodyRotationTarget.transform.rotation = Quaternion.identity;

        //Legs:rigth leg
        RHipTarget = new GameObject("RHipTarget");
        RHipTarget.transform.SetParent(BodyTarget.transform, false);
        RHipTarget.transform.rotation = Quaternion.identity;

        RKneeTarget = new GameObject("RKneeTarget");
        RKneeTarget.transform.SetParent(RHipTarget.transform, false);
        RKneeTarget.transform.rotation = Quaternion.identity;

        RFootTarget = new GameObject("RFootTarget");
        RFootTarget.transform.SetParent(RKneeTarget.transform, false);
        RFootTarget.transform.rotation = Quaternion.identity;


        //Legs:left leg
        LHipTarget = new GameObject("LHipTarget");
        LHipTarget.transform.SetParent(BodyTarget.transform, false);
        LHipTarget.transform.rotation = Quaternion.identity;

        LKneeTarget = new GameObject("LKneeTarget");
        LKneeTarget.transform.SetParent(LHipTarget.transform, false);
        LKneeTarget.transform.rotation = Quaternion.identity;

        LFootTarget = new GameObject("LFootTarget");
        LFootTarget.transform.SetParent(LKneeTarget.transform, false);
        LFootTarget.transform.rotation = Quaternion.identity;

        //Hands: right hand
        RShoulderTarget = new GameObject("RShoulderTarget");
        RShoulderTarget.transform.SetParent(BodyRotationTarget.transform, false);
        RShoulderTarget.transform.rotation = Quaternion.identity;

        RElbowTarget = new GameObject("RElbowTarget");
        RElbowTarget.transform.SetParent(RShoulderTarget.transform, false);
        RElbowTarget.transform.rotation = Quaternion.identity;

        RHandTarget = new GameObject("RHandTarget");
        RHandTarget.transform.SetParent(RElbowTarget.transform, false);
        RHandTarget.transform.rotation = Quaternion.identity;

        //Hands: left hand
        LShoulderTarget = new GameObject("LShoulderTarget");
        LShoulderTarget.transform.SetParent(BodyRotationTarget.transform, false);
        LShoulderTarget.transform.rotation = Quaternion.identity;

        LElbowTarget = new GameObject("LElbowTarget");
        LElbowTarget.transform.SetParent(LShoulderTarget.transform, false);
        LElbowTarget.transform.rotation = Quaternion.identity;

        LHandTarget = new GameObject("LHandTarget");
        LHandTarget.transform.SetParent(LElbowTarget.transform, false);
        LHandTarget.transform.rotation = Quaternion.identity;


        try{
            InitTCP();
            // button.onClick.Invoke();
        }
        catch (Exception e) {
            Debug.Log("On client connect exception " + e);
        }
    }

    private void InitTCP()
    {
        try {    
            receiveThread = new Thread (new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();
        }
        catch (Exception e) { 			
			Debug.Log("On client connect exception " + e); 		
		} 	
    }

    private void ReceiveData()
    {
        try {           
            listener = new TcpListener(IPAddress.Parse("127.0.0.1"), port);
            listener.Start();
            Byte[] bytes = new Byte[1024];

            while (true) {
                using (client = listener.AcceptTcpClient()) {
                    using (NetworkStream stream = client.GetStream()) {

                        int length;
                        while ((length = stream.Read(bytes, 0, bytes.Length)) != 0) {

                            var incommingData = new byte[length];
                            Array.Copy(bytes, 0, incommingData, 0, length);
                            string clientMessage = Encoding.ASCII.GetString(incommingData);
                            string[] res = clientMessage.Split(' ');

                            // Parse data
                            AssPos = new Vector3( float.Parse(res[0])
                                                  ,float.Parse(res[1])
                                                  ,float.Parse(res[2])
                            );
                            NeckPos = new Vector3( float.Parse(res[3])
                                                  ,float.Parse(res[4])
                                                  ,float.Parse(res[5])
                            );
                            ChestPos = new Vector3( float.Parse(res[6])
                                                  ,float.Parse(res[7])
                                                  ,float.Parse(res[8])
                            );
                            RHipPos = new Vector3( float.Parse(res[9])
                                                  ,float.Parse(res[10])
                                                  ,float.Parse(res[11])
                            );
                            RKneePos = new Vector3( float.Parse(res[12])
                                                  ,float.Parse(res[13])
                                                  ,float.Parse(res[14])
                            );
                            RAnklePos = new Vector3( float.Parse(res[15])
                                                  ,float.Parse(res[16])
                                                  ,float.Parse(res[17])
                            );
                            LHipPos = new Vector3( float.Parse(res[18])
                                                  ,float.Parse(res[19])
                                                  ,float.Parse(res[20])
                            );
                            LKneePos = new Vector3( float.Parse(res[21])
                                                  ,float.Parse(res[22])
                                                  ,float.Parse(res[23])
                            );
                            LAnklePos = new Vector3( float.Parse(res[24])
                                                  ,float.Parse(res[25])
                                                  ,float.Parse(res[26])
                            );
                            RShoulderPos = new Vector3( float.Parse(res[27])
                                                  ,float.Parse(res[28])
                                                  ,float.Parse(res[29])
                            );
                            RForeArmPos = new Vector3( float.Parse(res[30])
                                                  ,float.Parse(res[31])
                                                  ,float.Parse(res[32])
                            );
                            RWristPos = new Vector3( float.Parse(res[33])
                                                  ,float.Parse(res[34])
                                                  ,float.Parse(res[35])
                            );
                            LShoulderPos = new Vector3( float.Parse(res[36])
                                                  ,float.Parse(res[37])
                                                  ,float.Parse(res[38])
                            );
                            LForeArmPos = new Vector3( float.Parse(res[39])
                                                  ,float.Parse(res[40])
                                                  ,float.Parse(res[41])
                            );
                            LWristPos = new Vector3( float.Parse(res[42])
                                                  ,float.Parse(res[43])
                                                  ,float.Parse(res[44])
                            );
                        
                        }
                    }
                }
            }
        } catch(Exception e) {
            print (e.ToString());
        }
    }

    // Update is called once per frame
    void Update()
    {  

        // body rotatation and position
        transform.position = AssPos;
        
        BodyTarget.transform.position = AssPos;
        BodyTarget.transform.localRotation = Quaternion.Euler(0, HeadAngle_y, 0);
        BodyRotationTarget.transform.localPosition = ChestPos;
        BodyRotationTarget.transform.localRotation = Quaternion.Euler(0, BodyAngle_y, 0);
        
        NeckTarget.transform.localPosition = NeckPos;
        NeckTarget.transform.localRotation = Quaternion.Euler(0, 180, 0);

        // Feet
        RHipTarget.transform.localPosition = RHipPos;
        RKneeTarget.transform.localPosition = RKneePos;
        RFootTarget.transform.localPosition = RAnklePos;  
        LHipTarget.transform.localPosition = LHipPos;
        LKneeTarget.transform.localPosition = LKneePos;
        LFootTarget.transform.localPosition = LAnklePos;

        // Hands
        RShoulderTarget.transform.localPosition = RShoulderPos;
        LShoulderTarget.transform.localPosition = LShoulderPos;
        RElbowTarget.transform.localPosition = RForeArmPos; 
        LElbowTarget.transform.localPosition = LForeArmPos; 
        RHandTarget.transform.localPosition = RWristPos; 
        LHandTarget.transform.localPosition = LWristPos;

        animator.Update(1.0f);
    }

    void OnAnimatorIK(int layerIndex)
    {
        // set weigth to IK rotation and position
        animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 1.0f);
        animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 1.0f);
        animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, 1.0f);
        animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, 1.0f);
        animator.SetIKRotationWeight(AvatarIKGoal.RightFoot, 1.0f);
        animator.SetIKPositionWeight(AvatarIKGoal.RightFoot, 1.0f);
        animator.SetIKRotationWeight(AvatarIKGoal.LeftFoot, 1.0f);
        animator.SetIKPositionWeight(AvatarIKGoal.LeftFoot, 1.0f);
        animator.SetIKHintPositionWeight(AvatarIKHint.RightElbow, 1.0f);
        animator.SetIKHintPositionWeight(AvatarIKHint.LeftElbow, 1.0f);
        animator.SetIKHintPositionWeight(AvatarIKHint.RightKnee, 1.0f);
        animator.SetIKHintPositionWeight(AvatarIKHint.LeftKnee, 1.0f);

        // set up IK goals for hands and feet
        animator.SetIKRotation(AvatarIKGoal.RightHand, RHandTarget.transform.rotation); 
        animator.SetIKPosition(AvatarIKGoal.RightHand, RHandTarget.transform.position);
        animator.SetIKRotation(AvatarIKGoal.LeftHand, LHandTarget.transform.rotation);
        animator.SetIKPosition(AvatarIKGoal.LeftHand, LHandTarget.transform.position);
        animator.SetIKRotation(AvatarIKGoal.RightFoot, RFootTarget.transform.rotation);
        animator.SetIKPosition(AvatarIKGoal.RightFoot, RFootTarget.transform.position);
        animator.SetIKRotation(AvatarIKGoal.LeftFoot, LFootTarget.transform.rotation);
        animator.SetIKPosition(AvatarIKGoal.LeftFoot, LFootTarget.transform.position);

        // set up IK hints for elbows and knees
        animator.SetIKHintPosition(AvatarIKHint.RightElbow, RElbowTarget.transform.position);
        animator.SetIKHintPosition(AvatarIKHint.LeftElbow, LElbowTarget.transform.position);
        animator.SetIKHintPosition(AvatarIKHint.RightKnee, RKneeTarget.transform.position);
        animator.SetIKHintPosition(AvatarIKHint.LeftKnee, LKneeTarget.transform.position);
        

        // animator.SetIKPositionWeight(AvatarIKGoal.Neck, 1.0f);
        // animator.SetIKHintPositionWeight(AvatarIKHint.Chest, 1.0f);
        // animator.SetIKPosition(AvatarIKGoal.Neck, NeckTarget.transform.position);
        // animator.SetIKHintPosition(AvatarIKGoal.Chest, BodyRotationTarget.transform.position);
    }

    void OnApplicationQuit()
    {
        try
        {
            client.Close();
            // button.onClick.Invoke();
        }
        catch(Exception e)
        {
            Debug.Log(e.Message);
        }

        try
        {
            listener.Stop();
        }
        catch(Exception e)
        {
            Debug.Log(e.Message);
        }
    }
}


// public class BScreen : MonoBehaviour
// {
//     private bool recFlag;
//     private int widthF;
//     private int heightF;
//     private string filePath;
//     private string fileName;
//     private MediaEncoder encoder;
//     private int frameCnt;

//     public void NewStart() {
//         this.recFlag = false;
//         this.widthF = 770;  //320 //774
//         this.heightF = 340; //200 //347
//         // this.filePath = @"G:\Other\3Dfuck\1_PC\Screenshots\";
//         this.filePath = @"~/NewProjects/gdfjvdkfuv/movies";
//         this.fileName = @"my_movie.mp4";
//         this.frameCnt = 0;
//         Debug.Log(this.filePath);
//     }

//     IEnumerator RecordFrame()
//     {
//         yield return new WaitForEndOfFrame();
//         Texture2D texture = new Texture2D(this.widthF, this.heightF, TextureFormat.RGBA32, false);
//         Texture2D tex = new Texture2D(this.widthF, this.heightF, TextureFormat.RGBA32, false);
//         texture = ScreenCapture.CaptureScreenshotAsTexture();
//         // do something with texture

//         //texture.Resize(this.widthF,this.heightF);
//         for (int y = 0; y < this.heightF; y++)
//         {
//             for (int x = 0; x < this.widthF; x++)
//             {
//                 tex.SetPixel(x,y,texture.GetPixel(x,y));
//             }
//         }
//         //tex.Apply();
//         this.frameCnt++;
//         Debug.Log("Frame!" + this.frameCnt);
//         this.encoder.AddFrame(tex);
//         // cleanup
//         // Object.Destroy(tex);
//         // Object.Destroy(texture);
//     }

//     public void ScreenRec()
//     {
//         Debug.Log("first point");
//         if(!this.recFlag)
//         {
//             Debug.Log(this.filePath);
//             var encodedFilePath = Path.Combine(Path.GetDirectoryName(this.filePath),this.fileName);
//             Debug.Log("second point");
//                 var videoAttr = new VideoTrackAttributes
//                 {
//                     frameRate = new MediaRational(30),
//                     width = (uint)widthF,
//                     height = (uint)heightF,
//                     includeAlpha = false
//                 };
//                 Debug.Log("3d point");
//                 this.encoder = new MediaEncoder(encodedFilePath, videoAttr);
//                 Debug.Log("4 point");
//         }
//         else
//         {
//             this.encoder.Dispose();
//         }
//         this.recFlag = !this.recFlag;
//     }

//     public void LateUpdate()
//     {
//         if(this.recFlag)
//         {
//             StartCoroutine(RecordFrame());
//         }
//     }
// }