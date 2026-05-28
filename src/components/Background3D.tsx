import { Canvas, useFrame } from '@react-three/fiber';
import { useRef, useMemo } from 'react';
import * as THREE from 'three';

const BLUE = '#5b9dff';
const PURPLE = '#a78bfa';
const SILVER = '#cbd5e1';

function FloatingShapes() {
  const groupRef = useRef<THREE.Group>(null);
  const shapes = useMemo(() =>
    Array.from({ length: 10 }, (_, i) => ({
      position: [
        (Math.random() - 0.5) * 22,
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 10 - 6,
      ] as [number, number, number],
      scale: Math.random() * 0.5 + 0.2,
      speed: Math.random() * 0.4 + 0.15,
      rotSpeed: Math.random() * 0.01 + 0.002,
      color: i % 3 === 0 ? BLUE : i % 3 === 1 ? PURPLE : SILVER,
    })), []
  );

  useFrame((state) => {
    if (!groupRef.current) return;
    groupRef.current.children.forEach((child, i) => {
      const s = shapes[i];
      child.rotation.x += s.rotSpeed;
      child.rotation.y += s.rotSpeed * 1.2;
      child.position.y = s.position[1] + Math.sin(state.clock.elapsedTime * s.speed) * 0.6;
    });
  });

  return (
    <group ref={groupRef}>
      {shapes.map((shape, i) => (
        <mesh key={i} position={shape.position} scale={shape.scale}>
          <icosahedronGeometry args={[1, 0]} />
          <meshStandardMaterial
            color={shape.color}
            emissive={shape.color}
            emissiveIntensity={0.25}
            transparent
            opacity={0.28}
            wireframe
          />
        </mesh>
      ))}
    </group>
  );
}

function Particles() {
  const dotsRef = useRef<THREE.InstancedMesh>(null);
  const count = 70;
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const positions = useMemo(() =>
    Array.from({ length: count }, () => ({
      x: (Math.random() - 0.5) * 26,
      y: (Math.random() - 0.5) * 15,
      z: (Math.random() - 0.5) * 8 - 4,
      speed: Math.random() * 0.25 + 0.08,
    })), []
  );

  useFrame((state) => {
    if (!dotsRef.current) return;
    positions.forEach((pos, i) => {
      dummy.position.set(
        pos.x + Math.sin(state.clock.elapsedTime * pos.speed + i) * 0.25,
        pos.y + Math.cos(state.clock.elapsedTime * pos.speed + i) * 0.25,
        pos.z
      );
      dummy.scale.setScalar(0.04 + Math.sin(state.clock.elapsedTime * 1.5 + i) * 0.015);
      dummy.updateMatrix();
      dotsRef.current!.setMatrixAt(i, dummy.matrix);
    });
    dotsRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={dotsRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial color={SILVER} emissive={SILVER} emissiveIntensity={0.8} transparent opacity={0.7} />
    </instancedMesh>
  );
}

function GridFloor() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -5.5, -5]}>
      <planeGeometry args={[44, 44, 44, 44]} />
      <meshStandardMaterial
        color={BLUE}
        wireframe
        transparent
        opacity={0.05}
      />
    </mesh>
  );
}

export default function Background3D() {
  return (
    <div className="fixed inset-0 -z-10">
      <Canvas camera={{ position: [0, 0, 8], fov: 60 }} dpr={[1, 1.5]}>
        <ambientLight intensity={0.25} />
        <pointLight position={[10, 10, 10]} color={BLUE} intensity={0.5} />
        <pointLight position={[-10, -10, 5]} color={PURPLE} intensity={0.35} />
        <FloatingShapes />
        <Particles />
        <GridFloor />
      </Canvas>
    </div>
  );
}
