using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class gesture_bullet : MonoBehaviour
{

    public float speed = 25f;
    public GameObject ImpactEffect;
    public Rigidbody2D Rigid;
    public int Damage = 40;
    //public float Size = 1;

    // Start is called before the first frame update
    void Start()
    {
        Rigid = GetComponent<Rigidbody2D>();

        //transform.localScale = new Vector3(Size, Size, 1);

        Rigid.velocity = transform.right * speed;

        Invoke("DestroyBullet", 1f);        //1秒后子弹自动摧毁，以免占用内存
    }

    private void OnTriggerEnter2D(Collider2D hitInfo)
    {

        EnemyCrab enemyCrab = hitInfo.GetComponent<EnemyCrab>();

        if (enemyCrab != null)
        {
            enemyCrab.TakeDamage(Damage);
            GameObject.Find("Player").GetComponent<Weapon>().RestoreEnergy(10);           //给武器充能
            Instantiate(ImpactEffect, transform.position, transform.rotation);
            Destroy(gameObject);                //命中后摧毁子弹
        }

        EnemyJumper enemyJumper = hitInfo.GetComponent<EnemyJumper>();

        if (enemyJumper != null)
        {
            enemyJumper.TakeDamage(Damage);
            GameObject.Find("Player").GetComponent<Weapon>().RestoreEnergy(10);
            Instantiate(ImpactEffect, transform.position, transform.rotation);
            Destroy(gameObject);                //命中后摧毁子弹
        }

        EnemyOctopus enemyOctopus = hitInfo.GetComponent<EnemyOctopus>();

        if (enemyOctopus != null)
        {
            enemyOctopus.TakeDamage(Damage);
            GameObject.Find("Player").GetComponent<Weapon>().RestoreEnergy(10);
            Instantiate(ImpactEffect, transform.position, transform.rotation);
            Destroy(gameObject);                //命中后摧毁子弹
        }
             
    }

    void DestroyBullet()
    {
        Destroy(gameObject);
    }
}
