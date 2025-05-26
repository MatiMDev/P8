using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class TopDownCameraStreamer : MonoBehaviour
{
    [Header("Camera & Output")]
    public Camera topDownCamera;
    public RenderTexture renderTexture;

    [Header("Streaming Settings")]
    public string serverUrl = "http://127.0.0.1:5000/frame-bev";
    public int streamIntervalMS = 33; // ~30 FPS
    public int jpegQuality = 85;

    private WaitForSeconds streamDelay;

    void Start()
    {
        streamDelay = new WaitForSeconds(streamIntervalMS / 1000f);
        StartCoroutine(SendFramesLoop());
    }

    IEnumerator SendFramesLoop()
    {
        while (true)
        {
            yield return streamDelay;
            yield return SendFrame();
        }
    }

    IEnumerator SendFrame()
    {
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;

        Texture2D image = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);

        topDownCamera.targetTexture = renderTexture;
        topDownCamera.Render();

        image.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        image.Apply();

        byte[] jpgData = image.EncodeToJPG(jpegQuality);
        Destroy(image);

        RenderTexture.active = currentRT;

        UnityWebRequest request = UnityWebRequest.Put(serverUrl, jpgData);
        request.method = UnityWebRequest.kHttpVerbPOST;
        request.SetRequestHeader("Content-Type", "image/jpeg");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
            Debug.LogWarning("Top-down frame failed: " + request.error);
    }
}
