using UnityEngine;

public class CarController : MonoBehaviour
{
    public float moveSpeed = 10f;
    public float turnSpeed = 50f;
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        float moveInput = Input.GetAxis("Vertical");    // Forward/Backward
        float turnInput = Input.GetAxis("Horizontal");  // Left/Right

        // Move forward/backward
        Vector3 move = transform.forward * moveInput * moveSpeed * Time.fixedDeltaTime;
        rb.MovePosition(rb.position + move);

        // Invert steering direction if reversing
        if (Mathf.Abs(moveInput) > 0.01f)
        {
            float directionMultiplier = moveInput > 0 ? 1f : -1f;
            float turn = turnInput * turnSpeed * Time.fixedDeltaTime * directionMultiplier;

            Quaternion turnRotation = Quaternion.Euler(0, turn, 0);
            rb.MoveRotation(rb.rotation * turnRotation);
        }
    }
}
