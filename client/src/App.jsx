import React, { useState, useEffect } from "react";
import { Container, Form, Image, Button } from "react-bootstrap";
import styles from "./App.module.css";

import shakespeare from "./assets/shakespeare.jpeg";
import coolShakes from "./assets/cool-shakes.jpeg";

function App() {
  const [translation, setTranslation] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  function getTranslation(input) {
    console.log("Translating" + input);
      setTranslation("")
    setLoading(true);
    fetch(`/api/tr/${input}`)
      .then((res) => res.json())
      .then((data) => {
        setTranslation(data);
        setLoading(false);
      });
  }

  return (
    <Container fluid className={styles.wrapper}>
      <Form>
        <Container className={styles.formWrapper}>
          <Container className={styles.textboxWrapper}>
            <div className={styles.portraitWrapper}>
              <Image className={styles.portrait} src={shakespeare} rounded />
              <p className={`${styles.old} ${styles.name}`}>Old Shakespeare</p>
            </div>
            <Form.Group>
              <Form.Control
                className={`${styles.old} ${styles.textbox}`}
                as="textarea"
                rows={10}
                onChange={(e) => {
                  setText(e.target.value);
                  console.log(text);
                }}
              />
            </Form.Group>
          </Container>

          <Container className={styles.btnWrapper}>
            <Button
              className={styles.translateBtn}
              variant="primary"
              disabled={loading}
              onClick={() => getTranslation(text)}
            >
              {loading ? "Translating..." : "Translate"}
            </Button>
          </Container>

          <Container className={styles.textboxWrapper}>
            <Form.Group>
              <Form.Control
                className={`${styles.new} ${styles.textbox}`}
                as="textarea"
                readOnly
                value={translation}
                rows={20}
              />
            </Form.Group>
            <div className={styles.portraitWrapper}>
              <Image className={styles.portrait} src={coolShakes} rounded />
              <p className={`${styles.new} ${styles.name}`}>
                Modern Shakespeare
              </p>
            </div>
          </Container>
        </Container>
      </Form>
    </Container>
  );
}

export default App;
