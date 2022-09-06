using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;

using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;


public class gesture_weapon : MonoBehaviour
{
    public Transform FirePoint;
    public GameObject gesture_bullet;
    public CharacterController2D cc;
    public int charge;        //蓄力
    public int release;         //释放
    public float charge_time;    //蓄力时间
    public bool is_shooted;     //防止连射
    public string state;
    Process process = new Process();
    public bool shut;

    private string _ip = "127.0.0.1";
    private int _port = 444;
    private Socket _socket = null;
    private byte[] buffer = new byte[1024 * 1024 * 2];

    // Start is called before the first frame update
    void Start()
    {
        cc = GetComponent<CharacterController2D>();
        charge = 0;
        release = 0;
        charge_time = 1.0f;
        is_shooted = false;
        shut = false;

        string pyScriptPath = @"D:\EmergencyManagementSystem\venv\hand_detection_py3.7\gesture_3D.py";
        CallPythonBase(pyScriptPath);

        try
        {
            //1.0 实例化套接字(IP4寻找协议,流式协议,TCP协议)
            _socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            //2.0 创建IP对象
            IPAddress address = IPAddress.Parse(_ip);
            //3.0 创建网络端口,包括ip和端口
            IPEndPoint endPoint = new IPEndPoint(address, _port);
            //4.0 绑定套接字
            _socket.Bind(endPoint);
            //5.0 设置最大连接数
            _socket.Listen(int.MaxValue);
            // Console.WriteLine("监听{0}消息成功", _socket.LocalEndPoint.ToString());
            //6.0 开始监听
            Thread thread = new Thread(socket_receive);
            thread.Start();

        }
        catch (Exception ex)
        {
        }

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q))
        {
            process.Close();
            
            shut = true;
            
            Application.Quit();
        }
    }

    private void FixedUpdate()
    {
        if(state == "1")
        {
            is_shooted = false;
            if (charge == 0)
                charge += 1;
            else
                charge_time += 0.1f;
        }
        else if(state == "0")
        {
            if (release == 0)
                release += 1;
            else
            {
                if (!is_shooted)
                {
                    Shoot();
                    is_shooted = true;
                }
                release = 0;
                charge = 0;
                charge_time = 1.0f;
            }
        }                        
    }

    public void CallPythonBase(string pyScriptPath)
    {
        // python 的解释器位置 python.exe
        process.StartInfo.FileName = @"D:\EmergencyManagementSystem\venv\hand_detection_py3.7\Scripts\python.exe";

        UnityEngine.Debug.Log(pyScriptPath);

        process.StartInfo.UseShellExecute = false;
        process.StartInfo.Arguments = pyScriptPath;     // 路径+参数
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.RedirectStandardInput = true;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.CreateNoWindow = true;        // 不显示执行窗口

        // 开始执行，获取执行输出，添加结果输出委托
        process.Start();
    }

    private void socket_receive()
    {
        while(true)
        {
            Socket socket = _socket.Accept();
            // clientSocket.Send(Encoding.UTF8.GetBytes("服务端发送消息:"));
            Socket clientSocket = (Socket)socket;
            try
            {
                //获取从客户端发来的数据
                int length = clientSocket.Receive(buffer);
                state = Encoding.UTF8.GetString(buffer, 0, length);
                // UnityEngine.Debug.Log(state);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                clientSocket.Shutdown(SocketShutdown.Both);
                clientSocket.Close();
            }
            if(shut)
            { 
                clientSocket.Shutdown(SocketShutdown.Both);
                clientSocket.Close();
                break;
            }
        }
    }

    public void Shoot()
    {
        GameObject new_bullet = GameObject.Instantiate(gesture_bullet);
        new_bullet.transform.localScale = new Vector3(charge_time, charge_time, 1);
        Instantiate(new_bullet, FirePoint.position, FirePoint.rotation);
    }

    public void Exit()
    {
        Application.Quit();
    }
}
