import { Canvas, useFrame } from '@react-three/fiber';
import { useRef, useMemo } from 'react';
import * as THREE from 'three';

function FloatingCubes() {
  const groupRef = useRef<THREE.Group>(null);
  const cubes = useMemo(() => 
    Array.from({ length: 15 }, (_, i) => ({
      position: [
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 10 - 5,
      ] as [number, number, number],
      scale: Math.random() * 0.4 + 0.15,
      speed: Math.random() * 0.5 + 0.2,
      rotSpeed: Math.random() * 0.02 + 0.005,
      color: i % 3 === 0 ? '#00F5FF' : i % 3 === 1 ? '#BD00FF' : '#FFD700',
    })), []
  );

  useFrame((state) => {
    if (!groupRef.current) return;
    groupRef.current.children.forEach((child, i) => {
      const cube = cubes[i];
      child.rotation.x += cube.rotSpeed;
      child.rotation.y += cube.rotSpeed * 1.3;
      child.position.y = cube.position[1] + Math.sin(state.clock.elapsedTime * cube.speed) * 0.8;
    });
  });

  return (
    <group ref={groupRef}>
      {cubes.map((cube, i) => (
        <mesh key={i} position={cube.position} scale={cube.scale}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial
            color={cube.color}
            emissive={cube.color}
            emissiveIntensity={0.5}
            transparent
            opacity={0.6}
            wireframe={i % 2 === 0}
          />
        </mesh>
      ))}
    </group>
  );
}

function PacDots() {
  const dotsRef = useRef<THREE.InstancedMesh>(null);
  const count = 60;
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const positions = useMemo(() =>
    Array.from({ length: count }, () => ({
      x: (Math.random() - 0.5) * 24,
      y: (Math.random() - 0.5) * 14,
      z: (Math.random() - 0.5) * 8 - 4,
      speed: Math.random() * 0.3 + 0.1,
    })), []
  );

  useFrame((state) => {
    if (!dotsRef.current) return;
    positions.forEach((pos, i) => {
      dummy.position.set(
        pos.x + Math.sin(state.clock.elapsedTime * pos.speed + i) * 0.3,
        pos.y + Math.cos(state.clock.elapsedTime * pos.speed + i) * 0.3,
        pos.z
      );
      dummy.scale.setScalar(0.06 + Math.sin(state.clock.elapsedTime * 2 + i) * 0.02);
      dummy.updateMatrix();
      dotsRef.current!.setMatrixAt(i, dummy.matrix);
    });
    dotsRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={dotsRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial color="#FFD700" emissive="#FFD700" emissiveIntensity={1} />
    </instancedMesh>
  );
}

function GridFloor() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -5, -5]}>
      <planeGeometry args={[40, 40, 40, 40]} />
      <meshStandardMaterial
        color="#00F5FF"
        wireframe
        transparent
        opacity={0.08}
      />
    </mesh>
  );
}

export default function Background3D() {
  return (
    <div className="fixed inset-0 -z-10">
      <Canvas camera={{ position: [0, 0, 8], fov: 60 }} dpr={[1, 1.5]}>
        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} color="#00F5FF" intensity={0.5} />
        <pointLight position={[-10, -10, 5]} color="#BD00FF" intensity={0.3} />
        <FloatingCubes />
        <PacDots />
        <GridFloor />
      </Canvas>
    </div>
  );
}
