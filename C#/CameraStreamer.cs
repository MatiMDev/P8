using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class CameraStreamer : MonoBehaviour
{
    [Header("Camera & Output")]
    public Camera frontCamera;
    public RenderTexture renderTexture;

    [Header("Streaming Settings")]
    public string serverUrl = "http://127.0.0.1:5000/frame";
    public int streamIntervalMS = 33; // ~30 FPS
    public int jpegQuality = 85;

    private WaitForSeconds streamDelay;

    void Start()
    {
        // Convert milliseconds to seconds for WaitForSeconds
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
        // Backup current render target
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;

        // Create a Texture2D to read from RenderTexture
        Texture2D image = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);

        // Manually render the camera into the RenderTexture
        frontCamera.targetTexture = renderTexture;
        frontCamera.Render();

        image.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        image.Apply();

        // Encode image to JPG
        byte[] jpgData = image.EncodeToJPG(jpegQuality);
        Destroy(image);

        // Restore the active render target
        RenderTexture.active = currentRT;

        // Send HTTP POST
        UnityWebRequest request = UnityWebRequest.Put(serverUrl, jpgData);
        request.method = UnityWebRequest.kHttpVerbPOST;
        request.SetRequestHeader("Content-Type", "image/jpeg");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
            Debug.LogWarning("Frame send failed: " + request.error);
    }
}
